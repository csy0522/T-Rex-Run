# T-Rex, Run!

This project is about an automated Chrome Dino who is smart enough to jump over the obsticles (cactus). The Dino is trained using Asynchronous Advanced Actor-Critic algorithm (A3C) from Reinforcement Learning.

## Getting Started

Use any python IDE to open the project. I personally use Spyder from Anaconda.You can download both Anaconda or Spyder from the following links:
* [Anaconda](https://www.anaconda.com/distribution/) - The Data Science Platform for Python/R
* [Spyder](https://www.spyder-ide.org/) - The Scientific Python Development Environment

### Installation

Before running the project, type the following command to install the libraries that the project depends on

```
pip install numpy, matplotlib, keras, threading, opencv-contrib-python, gym, gym_chrome_dino, Pillow, selenium, tensorflow-gpu==1.15
```
Or simply type the following:

```
pip install -r requirements.txt
```

## Running the tests

- The description of each function is located on top of them. Please read them before running to understand the overall structure of the project. <br/>
- This project automates an agent (the Dino) in its environment (Desert, I assume, with cactuses) using Asynchronous Advantage Actor-Critic (A3C) Algorithm.<br/>
- Before testing, train the agent (Chrome dino) by running **train.py**. The training time depends on the number of threads and episodes. When done training, two .h5 type files, which are the trained actor and critic, will appear in the current directory.<br/>
- To run the test, go to terminal, and type:
```
python test.py <actor h5 file name> <critic h5 file name>
```
- The following is the trained agent/Dino jumping over cactuses:



![Dino Run~!](/data/Dino_Run.gif)

- The blue line is the actual stock price, and the red line is the prediction. <br/>
- For more detail, please read the descriptions on top of each function, and run **main.py**. Make sure to run it from an ide that's able to show graphs. The output will show more deails, including accuracy, loss, and more graphs.<br/>
- I also added a **ipynb** file for the main functionin in the **src** directory if you want to run it using [Jupyter Notebook](https://jupyter.org/)


## Deployment

Reinforcement Learning algorithms can be applied to any software agents in an environment. One of the best ways for beginners to approach Reinforcement Learning is learning Gym Environment and RL Algorithms. Click the following links to find out more about Reinforcement Learning.

* [Reinforcement Learning Basic](https://www.guru99.com/reinforcement-learning-tutorial.html) - Reinforcement Learning: What is, Algorithms, Applications, Example
* [OpenAI - Gym Environment](https://gym.openai.com/) - A Toolkit for Developing and Comparing Reinforcement Learning Algorithms
* [Reinforcement Learning Algorithms](https://gym.openai.com/) - list of algorithms of Reinforcement Learning

## Built With

* [Python](https://www.python.org/) - The Programming Language
* [Tensorflow](https://www.tensorflow.org/) - The End-to-end Open Source Machine Learning Platform
* [keras](https://keras.io/) - The Python Deep Learning Library
* [Gym](https://gym.openai.com/) - A Toolkit for Developing and Comparing Reinforcement Learning Algorithms
* [Gym_Chrome_Dino](https://pypi.org/project/gym-chrome-dino/) - An OpenAI Gym Environment for Chrome Dino / T-Rex Runner Game

## Author

* **CSY** - [csy0522](https://github.com/csy0522)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgements

* **Elvis Yu-Jing Lin** - [elvisyjlin](https://pypi.org/user/elvisyjlin/)


