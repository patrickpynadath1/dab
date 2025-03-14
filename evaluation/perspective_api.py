import collections
import json
import multiprocessing as mp
import time
from pathlib import Path
from typing import List, Union, Optional, Tuple, Dict, Any, Iterable

from googleapiclient import discovery
from googleapiclient.errors import HttpError
from tqdm.auto import tqdm

from typing import TypeVar, Iterable, List, Union, Any
import numpy as np
import torch
import os

# from utils.constants import PERSPECTIVE_API_ATTRIBUTES, PERSPECTIVE_API_KEY
# from utils.utils import batchify, load_cache

# PUT PERSPECTIVE API HERE
PERSPECTIVE_API_KEY = ""
PERSPECTIVE_API_ATTRIBUTES = (
    "TOXICITY",
    "SEVERE_TOXICITY",
    "IDENTITY_ATTACK",
    "INSULT",
    "THREAT",
    "PROFANITY",
    "SEXUALLY_EXPLICIT",
    "FLIRTATION",
)
PERSPECTIVE_API_ATTRIBUTES_LOWER = tuple(a.lower() for a in PERSPECTIVE_API_ATTRIBUTES)
T = TypeVar("T")


def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)


def batchify(data: Iterable[T], batch_size: int) -> Iterable[List[T]]:
    assert batch_size > 0

    batch = []
    for item in data:
        # Yield next batch
        if len(batch) == batch_size:
            yield batch
            batch = []

        batch.append(item)

    # Yield last un-filled batch
    if len(batch) != 0:
        yield batch


def load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
    with open(file) as f:
        for line in f:
            yield json.loads(line)


def load_cache(file: Path):
    if file.exists():
        with file.open() as f:
            for line in tqdm(f, desc=f"Loading cache from {file}"):
                yield json.loads(line)


def unpack_scores(response_json: dict) -> Optional[Tuple[dict, dict]]:
    if not response_json:
        return None

    attribute_scores = response_json["attributeScores"].items()

    summary_scores = {}
    span_scores = {}
    for attribute, scores in attribute_scores:
        attribute = attribute.lower()

        # Save summary score
        assert scores["summaryScore"]["type"] == "PROBABILITY"
        summary_scores[attribute] = scores["summaryScore"]["value"]

        # Save span scores
        for span_score_dict in scores["spanScores"]:
            assert span_score_dict["score"]["type"] == "PROBABILITY"
            span = (span_score_dict["begin"], span_score_dict["end"])
            span_scores.setdefault(span, {})[attribute] = span_score_dict["score"][
                "value"
            ]

    return summary_scores, span_scores


class PerspectiveAPI:
    def __init__(self, api_key: str = PERSPECTIVE_API_KEY, rate_limit: int = 60):
        self.service = self._make_service(api_key)
        self.last_request_time = -1  # satisfies initial condition
        self.rate_limit = rate_limit
        self.next_uid = 0

    def request(
        self, texts: Union[str, List[str]]
    ) -> List[Tuple[Optional[Dict[str, Any]], Optional[HttpError]]]:
        if isinstance(texts, str):
            texts = [texts]

        # Rate limit to 1 batch request per second
        assert len(texts) <= self.rate_limit
        time_since_last_request = time.time() - self.last_request_time
        time.sleep(10)
        # if time_since_last_request < 1:
        # print(time_since_last_request)
        # time.sleep(2.0 - time_since_last_request)
        self.last_request_time = time.time()

        # Keys guaranteed in insertion order (Python 3.7+)
        responses = {
            str(uid): None for uid in range(self.next_uid, self.next_uid + len(texts))
        }
        self.next_uid += len(texts)

        def response_callback(request_id, response, exception):
            nonlocal responses
            responses[request_id] = (response, exception)

        # Make API request
        batch_request = self.service.new_batch_http_request()
        for uid, text in zip(responses.keys(), texts):
            batch_request.add(
                self._make_request(text, self.service),
                callback=response_callback,
                request_id=uid,
            )
        batch_request.execute()

        return list(responses.values())

    def request_bulk(
        self,
        corpus: Union[Iterable[str], Iterable[Tuple[str, str]]],
        output_file: Union[str, Path],
        pbar: tqdm = None,
    ):
        # Check for output file
        output_file = Path(output_file)

        # Set up progress bar
        if not pbar:
            total = (
                len(corpus) if isinstance(corpus, collections.abc.Sequence) else None
            )
            pbar = tqdm(total=total, dynamic_ncols=True)
        pbar.set_description(f"Perspective API")

        i = 0
        num_failures = 0
        with output_file.open("w") as f:
            for batch in batchify(corpus, self.rate_limit):
                request_ids = None
                if isinstance(batch[0], tuple):
                    request_ids, batch = zip(*batch)

                for j, (response, exception) in enumerate(self.request(batch)):
                    response_dict = {
                        "request_id": request_ids[j] if request_ids else i,
                        "response": response,
                        "error": str(exception) if exception else None,
                    }

                    # Save response
                    f.write(json.dumps(response_dict))
                    f.write("\n")
                    f.flush()
                    os.fsync(f.fileno())
                    if exception:
                        print(exception)
                        num_failures += 1

                i += len(batch)
                pbar.update(len(batch))
                pbar.set_postfix(failures=num_failures, rate_limt=self.rate_limit)

    @staticmethod
    def _make_service(api_key: str):
        # Generate API client object dynamically based on service name and version
        # return discovery.build('commentanalyzer', 'v1alpha1', developerKey=api_key)
        client = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=api_key,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False,
        )
        return client

    @staticmethod
    def _make_request(text: str, service):
        analyze_request = {
            "comment": {"text": text},
            "requestedAttributes": {attr: {} for attr in PERSPECTIVE_API_ATTRIBUTES},
            "spanAnnotations": True,
            "languages": ["en"],
        }
        return service.comments().analyze(body=analyze_request)


class PerspectiveWorker:
    SENTINEL = "STOP"

    def __init__(self, out_file: Path, total: int, rate_limit: int):
        if not rate_limit:
            print("Disabling Perspective API (rps is 0)")
            self.enabled = False
            return
        self.enabled = True

        self.requests_handled = set()
        for response in load_cache(out_file):
            print(response)
            self.requests_handled.add(response["request_id"])
        total -= len(self.requests_handled)

        # Setup worker thread
        self.task_queue = mp.Queue()
        self.process = mp.Process(
            target=self.perspective_worker,
            args=(self.task_queue, out_file, total, rate_limit),
        )
        self.process.start()

    def __call__(self, request_id: str, text: str):
        if not self.enabled:
            return

        if request_id not in self.requests_handled:
            self.task_queue.put((request_id, text))

    def stop(self):
        if not self.enabled:
            return

        print("Waiting for Perspective to finish...")
        self.task_queue.put(self.SENTINEL)
        self.process.join()

    @classmethod
    def perspective_worker(
        cls, queue: mp.Queue, responses_file: Path, total: int, rate_limit: int
    ):
        queue_iter = iter(queue.get, cls.SENTINEL)
        api = PerspectiveAPI(rate_limit=rate_limit)
        with tqdm(total=total, dynamic_ncols=True, position=1) as pbar:
            api.request_bulk(queue_iter, output_file=responses_file, pbar=pbar)


def test_perspective_api():
    api = PerspectiveAPI()

    text_success = "Testing"
    text_error = "x" * (20480 + 1)

    score_1, error_1 = api.request(text_success)[0]
    assert score_1 and not error_1

    score_2, error_2 = api.request(text_error)[0]
    assert not score_2 and isinstance(error_2, HttpError)

    multi_score, multi_error = zip(*api.request([text_success, text_error]))
    assert multi_score == (score_1, score_2)
    assert tuple(map(str, multi_error)) == tuple(map(str, (error_1, error_2)))
