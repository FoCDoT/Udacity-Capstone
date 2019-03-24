# Deep descriptive answer evaluation
Given a short descriptive answer can we give it the correct score?


dependencies are freezed in requirenment.txt


## Running the models

The datasets should be a csv document with the format given in `data` directory.
By default all the models save their weights in a _checkpoints_ folder in their corresponding directory

#### To train a model
In the corresponding model directory run the following command

`python main.py`

#### To test and get predictions
In the corresponding model directory run the following command

`python main.py --test`

the above command will generate an _outputs.csv_ file containing predictions.

#### To evaluate against the kappa metric

 the _outputs.csv_ file generated by training file to see the kappa score generated

`python evaluate.py`
