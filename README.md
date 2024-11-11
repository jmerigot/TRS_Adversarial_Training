# Transferability Reduced Smooth (TRS) Ensemble Training for Adversarial Transferability Attacks

This group project was completed for the Data Science Lab course as part of the IASD Master Program (AI Systems and Data Science) at PSL Research University during Semester 1 of the 2023-24 academic year. The full report associated with this project can be found in the *report.pdf* file of this repository, which details our approach and methods used to complete this project, as well as the analyzed results.

**Problem Statement:** How to mitigate transferability of adversarial attacks across different models by employing Transferability Reduced Smooth (TRS) ensemble training on the CIFAR-10 dataset.

**Our Approach:** This project addresses the challenge of adversarial attack transferability in machine learning, where deceptive inputs designed for one model can mislead others. Our aim is to investigate the Transferability Reduced Smooth (TRS) ensemble training method to reduce this transferability, thereby enhancing the robustness of our neural network models against such attacks. We first adversarially trained models on various attacks on the CIFAR-10 dataset to serve as a baseline for our later method. We then delved into the world of ensemble robustness.


## General Information: Basic usage

Install python dependencies with pip: 

    $ pip install -r requirements.txt

Test the basic model:

    $ ./model.py
    Testing with model from 'models/default_model.pth'. 
    Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to ./data/cifar-10-python.tar.gz
    100.0%
    Extracting ./data/cifar-10-python.tar.gz to ./data/
    Model natural accuracy (test): 53.07

(Re)train the basic model:

    $ ./model.py --force-train
    Training model
    models/default_model.pth
    Files already downloaded and verified
    Starting training
    [1,   500] loss: 0.576
    [1,  1000] loss: 0.575
    ...

Train/test the basic model and store the weights to a different file:

    $ ./model.py --model-file models/mymodel.pth
    ...

Load the module project and test it as close as it will be tested on the testing plateform:

    $ ./test_project.py

Even safer: do it from a different directory:

    $ mkdir tmp
    $ cd /tmp
    $ ../test_project.py ../

### Modifying the project

You can modify anything inside this git repository, it will work as long as:

- it contains a `model.py` file in the root directory
- the `model.py` file contains a class called `Net` derived from `torch.nn.Module`
- the `Net` class has a function call `load_for_testing()` that initializes the model for testing (typically by setting the weights properly).  The default load_for_testing() loads and store weights from a model file, you will also need to make sure the repos contains a model file that can be loaded into the `Net` architecture using Net.load(model_file).
- You may modify this `README.md` file. 

### Before pushing

When you have made improvements your version of the git repository:

1. Add and commit every new/modified file to the git repository, including your model files in models/.(Check with `git status`) *DO NOT CHECK THE DATA IN PLEASE!!!!*
2. Run `test_project.py` and verify the default model file used by load_for_testing() is the model file that you actually want to use for testing on the platform. 
3. Push your last change

Note: If you want to avoid any problems, it is a good idea to make a local copy of your repos (with `git clone <repos> <repos-copy>`) and to test the project inside this local copy.


## Acknowledgements
This project was made possible with the guidance and support of the following :
 
- **Prof. Benjamin Negrevergne**
  - Associate professor at the Lamsade laboratory from PSL – Paris Dauphine university, in Paris (France)
  - Active member of the *MILES* project
  - Co-director of the IASD Master Program (AI Systems and Data Science) with Olivier Cappé.

- **Alexandre Verine**
  - PhD candidate at LAMSADE, a joint research lab of Université Paris-Dauphine and Université PSL, specializing in Machine Learning.

This project was a group project and was accomplished through team work involving the following students :

- **Thomas Boudras**
  - Masters student in the IASD Master Program (AI Systems and Data Science) at PSL Research University.

- **Vivien Conti**
  - Masters student in the IASD Master Program (AI Systems and Data Science) at PSL Research University.

- **Jules Merigot** (myself)
  - Masters student in the IASD Master Program (AI Systems and Data Science) at PSL Research University.

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

