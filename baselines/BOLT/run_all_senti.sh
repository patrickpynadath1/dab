#!bin/bash 

for senti in pos neg; 
do 
    for length in 12 20 50;
    do 
        python sentiment_generate_with_bias.py $length $senti
    done
done 