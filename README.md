# BFSegmenter

The BFSegmenter segments audio files and classifies each segment as background, foreground, or background with foreground. Additionaly, for each segment the affect is predicted on a scale of valence and arousal.

![Pipeline](/images/pipeline.png)

Sound designers and soundscape composers manually segment audio files into building blocks for use in a composition. **Machine learning (ridge regression)** is used to classify segments in an audio file automatically. The model has a **83.0% true positive classification rate**. 

Russel’s model
suggests all emotions are distributed in a circular space (https://psycnet.apa.org/record/1981-25062-001).
High levels of valence correspond to pleasant sounds while
low valence levels correspond to unpleasant sounds. Further, high levels of arousal correspond to exciting sounds while low levels correspond to calming sounds. **Levels of valence and arousal are quantified using machine learning for emotion prediction (random forest regression)**. The emotion prediction models use a subset of extracted features to predict valence and arousal on a scale from -1 to 1 for each segment in an audio file.

![Affect Accuracy](/images/affect_accuracy.png)

Example implimentation of the segmenter in *extract_audacity_labels.py*.

## Segment format

    bf type
    duration
    start
    end
    features
    arousal
    valence
    bf probabilities

## Dependancies
Essentia - an open-source library for tools for audio and music analysis, description and synthesis. https://essentia.upf.edu/ 

Scikit-learn - a free software machine learning library for the Python programming language.

For full requirements, check *requirements.txt*.

## Authors

 - Miles Thorogood
 - Joshua Kranabetter

## License

This project is licensed under the MIT License - see the *LICENSE* file for details.
