# nyt-caption-guesser

New York Times “What’s Going On in This Picture?”  
Description: My project is based on the NYT articles, “What’s Going On in This Picture?”. These articles come out weekly and your goal is to guess the caption of the image. This model's purpose is to guess the caption of an image.

<img src="image-url" alt="Alt Text" width="300" height="200">

![IMG_3256](https://github.com/user-attachments/assets/71ba6f42-adf2-44aa-a322-cf0184d9d003)

</img>

The Algorithm
I had to use many different .py files because I had to train, finetune, and also scrape images for my model. Inside my scraping.py file, I had to create a folder full of the html files of the articles. These articles held the image and the captions which I used to train my model. After gathering the data to train my model. I downloaded my model which is Qwen2-VL-2B-Instruct. Then I started training and fine tuning my model. I started by creating the file to pull the information from scraping. Once I finished doing that, I finetuned the model by giving it more data.

The Struggles
The struggles with doing this project was the amount of time it took to find a model, download it, train it, and test it. Finding the correct model to use and training took most of my time during this project. 
Running this project
First cd into the folder by typing “cd ./jetson-inference/python/training/classification/data/final_project” in the terminal
If you want choose the image you want to caption, you put an image inside the senticap_images folder Here is the path: /home/nvidia/jetson-inference/python/training/classification/data/final_project/archive/senticap_images
Once you placed the image in this folder, you go to line 14 where it says img_path and replace the jpg with your image name
If you want to run this project, you type python inference_git_captioning.py in the terminal
The immediate message that will pop up is “Using a slow image processor as `use_fast` is unset and a slow processor was saved with this model. `use_fast=True` will be the default behavior in v4.52, even if the model was saved with a slow processor. This will result in minor differences in outputs. You'll still be able to use a slow processor with `use_fast=False`.” and after a while your caption for your image will appear

https://drive.google.com/file/d/1udTtV23mKFHzHs22ZWGdSYO2OL2q1PFG/view?usp=sharing
