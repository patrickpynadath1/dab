/**
 * Chart data for DAB project website
 * Contains configuration for all experiment charts
 */
const chartData = {
    // SENTIMENT EXPERIMENT CHARTS
    
    // Control Metrics - Sentiment
    sentimentInternal: {
        labels: ["DAB (Ours)", "MuCoLA", "COLD", "BOLT", "LM-Steer"],
        datasetLabel: "Yelp Polarity Classifier (Higher is Better)",
        data: [89.4, 84.3, 51.5, 74.7, 90.0], 
        backgroundColor: [
        'rgba(255, 99, 132, 0.9)',  // DAB - highlighted
        'rgba(54, 162, 235, 0.7)',  // MuCoLA
        'rgba(255, 206, 86, 0.7)',  // COLD
        'rgba(75, 192, 192, 0.7)',  // BOLT
        'rgba(153, 102, 255, 0.7)'  // LM-Steer
        ],
        borderColor: [
        'rgb(255, 99, 132)',  // DAB
        'rgb(54, 162, 235)',  // MuCoLA
        'rgb(255, 206, 86)',  // COLD
        'rgb(75, 192, 192)',  // BOLT
        'rgb(153, 102, 255)'  // LM-Steer
        ],
        borderWidth: [3, 1, 1, 1, 1],  // Thicker border for DAB
        minScale: 50,
        maxScale: 100,
        stepSize: 10,
        tickSuffix: '%',
        yAxisTitle: 'Average Classifier Prediction (%)',
        xAxisTitle: 'Method'
    },
    sentimentExternal: {
        labels: ["DAB (Ours)", "MuCoLA", "COLD", "BOLT", "LM-Steer"],
        datasetLabel: "SST Polarity Classifier (Higher is Better)",
        data: [97.5, 89.9, 67.0, 87.8, 94.8], 
        backgroundColor: [
        'rgba(255, 99, 132, 0.9)',  // DAB - highlighted
        'rgba(54, 162, 235, 0.7)',  // MuCoLA
        'rgba(255, 206, 86, 0.7)',  // COLD
        'rgba(75, 192, 192, 0.7)',  // BOLT
        'rgba(153, 102, 255, 0.7)'  // LM-Steer
        ],
        borderColor: [
        'rgb(255, 99, 132)',  // DAB
        'rgb(54, 162, 235)',  // MuCoLA
        'rgb(255, 206, 86)',  // COLD
        'rgb(75, 192, 192)',  // BOLT
        'rgb(153, 102, 255)'  // LM-Steer
        ],
        borderWidth: [3, 1, 1, 1, 1],
        minScale: 60,
        maxScale: 100,
        stepSize: 10,
        tickSuffix: '%',
        yAxisTitle: 'External Classifier Accuracy (%)',
        xAxisTitle: 'Method'
    },
    
    // Fluency Metrics - Sentiment
    sentimentPerplexity: {
        labels: ["DAB (Ours)", "MuCoLA", "COLD", "BOLT", "LM-Steer"],
        datasetLabel: "Perplexity (Lower is Better)",
        data: [11.8, 34.8, 15.9, 9.9, 72.2],
        backgroundColor: [
        'rgba(255, 99, 132, 0.9)',  // DAB - highlighted
        'rgba(54, 162, 235, 0.7)',  // MuCoLA
        'rgba(255, 206, 86, 0.7)',  // COLD
        'rgba(75, 192, 192, 0.7)',  // BOLT
        'rgba(153, 102, 255, 0.7)'  // LM-Steer
        ],
        borderColor: [
        'rgb(255, 99, 132)',  // DAB
        'rgb(54, 162, 235)',  // MuCoLA
        'rgb(255, 206, 86)',  // COLD
        'rgb(75, 192, 192)',  // BOLT
        'rgb(153, 102, 255)'  // LM-Steer
        ],
        borderWidth: [3, 1, 1, 1, 1],
        minScale: 0,
        maxScale: 25,
        stepSize: 5,
        yAxisTitle: 'Perplexity Score',
        xAxisTitle: 'Method'
    },
    sentimentCola: {
        labels: ["DAB (Ours)", "MuCoLA", "COLD", "BOLT", "LM-Steer"],
        datasetLabel: "Acceptability (Higher is Better)",
        data: [86.0, 68.1, 73.1, 87.4, 56.4],
        backgroundColor: [
        'rgba(255, 99, 132, 0.9)',  // DAB - highlighted
        'rgba(54, 162, 235, 0.7)',  // MuCoLA
        'rgba(255, 206, 86, 0.7)',  // COLD
        'rgba(75, 192, 192, 0.7)',  // BOLT
        'rgba(153, 102, 255, 0.7)'  // LM-Steer
        ],
        borderColor: [
        'rgb(255, 99, 132)',  // DAB
        'rgb(54, 162, 235)',  // MuCoLA
        'rgb(255, 206, 86)',  // COLD
        'rgb(75, 192, 192)',  // BOLT
        'rgb(153, 102, 255)'  // LM-Steer
        ],
        borderWidth: [3, 1, 1, 1, 1],
        minScale: 50,
        maxScale: 100,
        stepSize: 5,
        tickSuffix: '%',
        yAxisTitle: 'CoLA Acceptability (%)',
        xAxisTitle: 'Method'
    },
    
    // TOXICITY EXPERIMENT CHARTS
    
    // Control Metrics - Toxicity
    toxicityInternal: {
        labels: ["DAB (Ours)", "MuCoLA", "COLD", "BOLT"],
        datasetLabel: "Internal Toxicity Score (Lower is Better)",
        data: [5.7, 9.8, 13.6, 6.5], 
        backgroundColor: [
        'rgba(255, 99, 132, 0.9)',  // DAB - highlighted
        'rgba(54, 162, 235, 0.7)',  // MuCoLA
        'rgba(255, 206, 86, 0.7)',  // COLD
        'rgba(75, 192, 192, 0.7)',  // BOLT
        ],
        borderColor: [
        'rgb(255, 99, 132)',  // DAB
        'rgb(54, 162, 235)',  // MuCoLA
        'rgb(255, 206, 86)',  // COLD
        'rgb(75, 192, 192)',  // BOLT
        ],
        borderWidth: [3, 1, 1, 1],
        minScale: 0,
        maxScale: 30,
        stepSize: 5,
        tickSuffix: '%',
        yAxisTitle: 'Internal Toxicity Score',
        xAxisTitle: 'Method'
    },
    toxicityExternal: {
        labels: ["DAB (Ours)", "MuCoLA", "COLD", "BOLT", "LM-Steer"],
        datasetLabel: "External Avg Max Toxicity (Lower is Better)",
        data: [21.1, 26.9, 26.6, 26.5, 21.1], 
        backgroundColor: [
        'rgba(255, 99, 132, 0.9)',  // DAB - highlighted
        'rgba(54, 162, 235, 0.7)',  // MuCoLA
        'rgba(255, 206, 86, 0.7)',  // COLD
        'rgba(75, 192, 192, 0.7)',  // BOLT
        'rgba(153, 102, 255, 0.7)'  // LM-Steer
        ],
        borderColor: [
        'rgb(255, 99, 132)',  // DAB
        'rgb(54, 162, 235)',  // MuCoLA
        'rgb(255, 206, 86)',  // COLD
        'rgb(75, 192, 192)',  // BOLT
        'rgb(153, 102, 255)'  // LM-Steer
        ],
        borderWidth: [3, 1, 1, 1, 1],
        minScale: 0,
        maxScale: 30,
        stepSize: 5,
        tickSuffix: '%',
        yAxisTitle: 'External Toxicity Score',
        xAxisTitle: 'Method'
    },
    
    // Fluency Metrics - Toxicity
    toxicityPerplexity: {
        labels: ["DAB (Ours)", "MuCoLA", "COLD", "BOLT", "LM-Steer"],
        datasetLabel: "Perplexity",
        data: [25.6, 58.0, 38.9, 27.3, 52.7],
        backgroundColor: [
        'rgba(255, 99, 132, 0.9)',  // DAB - highlighted
        'rgba(54, 162, 235, 0.7)',  // MuCoLA
        'rgba(255, 206, 86, 0.7)',  // COLD
        'rgba(75, 192, 192, 0.7)',  // BOLT
        'rgba(153, 102, 255, 0.7)'  // LM-Steer
        ],
        borderColor: [
        'rgb(255, 99, 132)',  // DAB
        'rgb(54, 162, 235)',  // MuCoLA
        'rgb(255, 206, 86)',  // COLD
        'rgb(75, 192, 192)',  // BOLT
        'rgb(153, 102, 255)'  // LM-Steer
        ],
        borderWidth: [3, 1, 1, 1, 1],
        minScale: 20,
        maxScale: 60,
        stepSize: 5,
        yAxisTitle: 'Perplexity Score (Lower is Better)',
        xAxisTitle: 'Method'
    },
    toxicityCola: {
        labels: ["DAB (Ours)", "MuCoLA", "COLD", "BOLT", "LM-Steer"],
        datasetLabel: "Acceptability (%)",
        data: [80.6, 69.1, 66.7, 83.0, 72.2],
        backgroundColor: [
        'rgba(255, 99, 132, 0.9)',  // DAB - highlighted
        'rgba(54, 162, 235, 0.7)',  // MuCoLA
        'rgba(255, 206, 86, 0.7)',  // COLD
        'rgba(75, 192, 192, 0.7)',  // BOLT
        'rgba(153, 102, 255, 0.7)'  // LM-Steer
        ],
        borderColor: [
        'rgb(255, 99, 132)',  // DAB
        'rgb(54, 162, 235)',  // MuCoLA
        'rgb(255, 206, 86)',  // COLD
        'rgb(75, 192, 192)',  // BOLT
        'rgb(153, 102, 255)'  // LM-Steer
        ],
        borderWidth: [3, 1, 1, 1, 1],
        minScale: 60,
        maxScale: 100,
        stepSize: 5,
        tickSuffix: '%',
        yAxisTitle: 'CoLA Acceptability (%)',
        xAxisTitle: 'Method'
    },
    
    // KEYWORD EXPERIMENT CHARTS
    
    // Control Metrics - Keyword
    keywordInclusion: {
        labels: ["DAB (Ours)", "MuCoLA", "COLD", "BOLT"],
        datasetLabel: "BertScore (Higher is Better)",
        data: [83.03, 80.83, 81.23, 82.93], 
        backgroundColor: [
        'rgba(255, 99, 132, 0.9)',  // DAB - highlighted
        'rgba(54, 162, 235, 0.7)',  // MuCoLA
        'rgba(255, 206, 86, 0.7)',  // COLD
        'rgba(75, 192, 192, 0.7)'   // BOLT
        ],
        borderColor: [
        'rgb(255, 99, 132)',  // DAB
        'rgb(54, 162, 235)',  // MuCoLA
        'rgb(255, 206, 86)',  // COLD
        'rgb(75, 192, 192)'   // BOLT
        ],
        borderWidth: [3, 1, 1, 1],
        minScale: 80,
        maxScale: 85,
        stepSize: 1,
        tickSuffix: '%',
        yAxisTitle: 'BertScore',
        xAxisTitle: 'Method'
    },
    keywordBertscore: {
        labels: ["DAB (Ours)", "MuCoLA", "COLD", "BOLT"],
        datasetLabel: "Inclusion Rate (Higher is Better)",
        data: [99.0, 100.0, 100.0, 99.1], 
        backgroundColor: [
        'rgba(255, 99, 132, 0.9)',  // DAB - highlighted
        'rgba(54, 162, 235, 0.7)',  // MuCoLA
        'rgba(255, 206, 86, 0.7)',  // COLD
        'rgba(75, 192, 192, 0.7)'   // BOLT
        ],
        borderColor: [
        'rgb(255, 99, 132)',  // DAB
        'rgb(54, 162, 235)',  // MuCoLA
        'rgb(255, 206, 86)',  // COLD
        'rgb(75, 192, 192)'   // BOLT
        ],
        borderWidth: [3, 1, 1, 1],
        minScale: 90,
        maxScale: 100,
        stepSize: 0.01,
        yAxisTitle: 'Inclusion Rate',
        xAxisTitle: 'Method'
    },
    
    // Fluency Metrics - Keyword
    keywordPerplexity: {
        labels: ["DAB (Ours)", "MuCoLA*", "COLD*", "BOLT"],
        datasetLabel: "Perplexity",
        data: [23.4, 475.3, 242, 32.0],
        backgroundColor: [
        'rgba(255, 99, 132, 0.9)',  // DAB - highlighted
        'rgba(54, 162, 235, 0.7)',  // MuCoLA
        'rgba(255, 206, 86, 0.7)',  // COLD
        'rgba(75, 192, 192, 0.7)'   // BOLT
        ],
        borderColor: [
        'rgb(255, 99, 132)',  // DAB
        'rgb(54, 162, 235)',  // MuCoLA
        'rgb(255, 206, 86)',  // COLD
        'rgb(75, 192, 192)'   // BOLT
        ],
        borderWidth: [3, 1, 1, 1],
        minScale: 20,
        maxScale: 40,
        stepSize: 5,
        yAxisTitle: 'Perplexity Score (Lower is Better)',
        xAxisTitle: 'Method'
    },
    keywordCola: {
        labels: ["DAB (Ours)", "MuCoLA", "COLD", "BOLT"],
        datasetLabel: "Acceptability (Higher is Better)",
        data: [72.6, 24.8, 20.5, 70.5], 
        backgroundColor: [
        'rgba(255, 99, 132, 0.9)',  // DAB - highlighted
        'rgba(54, 162, 235, 0.7)',  // MuCoLA
        'rgba(255, 206, 86, 0.7)',  // COLD
        'rgba(75, 192, 192, 0.7)'   // BOLT
        ],
        borderColor: [
        'rgb(255, 99, 132)',  // DAB
        'rgb(54, 162, 235)',  // MuCoLA
        'rgb(255, 206, 86)',  // COLD
        'rgb(75, 192, 192)'   // BOLT
        ],
        borderWidth: [3, 1, 1, 1],
        minScale: 0,
        maxScale: 100,
        stepSize: 5,
        tickSuffix: '%',
        yAxisTitle: 'CoLA Acceptability (%)',
        xAxisTitle: 'Method'
    }
};