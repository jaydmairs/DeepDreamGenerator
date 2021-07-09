# DeepDreamGenerator
## Visual detail generation tool

DeepDreamGeneration is a learning/research project that uses Tensorflow and Google's [Deep Dream] project to generate appropriate visual details for a simple input clipart, cartoon, or drawing.

- Uses a clipart file named "bulldog.jpg" as the input file.
- filter_images directory contains a photograph of a dog that's used to learn visual details.

## Usage

- Make sure the Tensorflow Python module is installed and working.
- Run script "deep_dream_new.py" in Spyder, iPython, or command line
- After a few seconds, an image window should appear with increasingly detailed views of the output image.  If you're using Spyder or iPython and the window doesn't appear, you can set Spyder or iPython up to display images in a separate window

## Output
The hardcoded example uses this clipart image as input:  
<img src="https://github.com/jaydmairs/DeepDreamGenerator/blob/main/bulldog.jpg?raw=true" width="500">  
After running, the output image will look similar to this:  
<img src="https://github.com/jaydmairs/DeepDreamGenerator/blob/main/generated_dog.png?raw=true" width="500">  

