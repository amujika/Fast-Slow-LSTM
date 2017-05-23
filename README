#The code is largely based on [1], [2] and [3]

-Files:

---FS_example.py: Minimal example (20 lines of code) of how to create
                  a FS-LSTM and run one step of it, using the Tensorflow LSTM.
                  
---main.py:       Main file which should be run to replicate the PTB
                  and enwik8 experiments.
                  To run PTB experiment:     python3 main.py --model ptb
                  To run enwik8 experiment:  python3 main.py --model enwik8
                  We only include the PTB dataset due to size limitations.
                  To run the enwik8 experiments the dataset should be 
                  added to "data/enwik8" divided in three files: "train",
                  "valid" and "test"

---config.py:     Hyper-parameters are loaded from here according
                  to which experiment is being performed. All hyper-
                  parameters from Table 3 of the paper can be set
                  using this file.

---reader.py:     Reads training, validation and testing data from
                  the data folder.

---LNLSTM.py:     Layer normalized LSTM with Zoneout.

---aux.py:        File with auxiliary functions for LNLSTM.py.

---FSRNN.py:      Fast-Slow RNN class that can be used with any
                  standard Tensorflow RNN. 



[1] https://github.com/tensorflow/models/tree/master/tutorials/rnn/ptb
[2] https://github.com/pbhatia243/tf-layer-norm
[3] https://github.com/hardmaru/supercell
