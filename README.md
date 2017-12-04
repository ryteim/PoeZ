# PoeZ
Haiku Generator

General usage:

> python lstm.py [-h] [-d DATA] -nw NETWORK_WEIGHTS -m {train,sample} [-q QUERY]

Example usage for training:

> python lstm.py -m train -d smaller_haiku.txt -nw temp_weights/test_weights

Example usage for sampling:

> python lstm.py -m sample -d smaller_haiku.txt -nw temp_weights/test_weights -q breeze
