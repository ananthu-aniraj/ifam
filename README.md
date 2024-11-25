# Trust me, I'm an ENGINEER: End-to-end input masks for guaranteed interpretability and enhanced robustness

Implementation of the paper "Trust me, I am an Engineer: End-to-end input masks for guaranteed interpretability and enhanced robustness" 

# Abstract
We introduce an interpretable-by-design method based on learned binary attention masks on the input image, which guarantees that only attended image regions have an impact on the prediction. Contextual elements in images can heavily influence object perception, potentially leading to biased representations, particularly when the object is in an out-of-distribution background setting. At the same time, many vision tasks, such as object detection, require context to be solved. To address this dilemma, our method employs a two-stage framework: the first stage receives the whole image and detects object parts. The second stage focuses only on the detected foreground image regions, which contain the relevant objects, effectively disregarding potentially spurious contextual cues. Both stages are trained jointly, allowing the second stage to provide feedback that further refines the first one. Results on four robustness benchmarks validate the usefulness of the approach.

# Setup
To install the required packages, run the following command:
```conda env create -f environment.yml```