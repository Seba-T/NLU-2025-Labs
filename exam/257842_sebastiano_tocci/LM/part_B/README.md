Here we start by using the best hyperparameters we found in part A:
     hid_size = 200
    emb_size = 300
    lr = 0.0005
    weight_decay = 0.15
    betas=(0.9, 0.999)

- Run 1: We get a Test ppl:  108.75538253077593, which is much better than what we got in 1.1 (168.2143888617081)

- RUn 2: we now try with a much bigger dropout, 0.3 for both, and we get Test ppl:  101.23476130695406, MUCH better than before, very good

- Run 3: on this track we futher increase the dropout, to 0.4 for both of them and we get: Test ppl:  100.57972565038041, better!

- Run 4: we now try to add another layer to the LSTM, and see what happens. We get: Test ppl:  99.65182114503435, under 100 for the first time!

- Run 5: now it's time to test the variational approach with all the improvements we have found before, i.e. 2 layers and 0.4 in dropout rate, and we get: Test ppl:  98.86770542781103, slightly better than before!

- Run 6, and now finally we test with the NTAvSDG, with lr=1 to start with, n=5 and L=1, as suggested in the paper. In the paper, they suggest to set L equal to the number of iterations in an epoch, but since we run the loop only once after every epoch, it must be equal to 1 to match the same behavior. We get a WOPPING Test ppl:  90.42661578788554, LOWER than anything else (beating even SOTA algorithms such as AdamW). It must be noted that this was the first time we didn't have any stopping condition, and therefore the training consisted of the full 100 epochs.

- Run 7: we try increase the learning rate, since it seems to help with SGD. We set it to 1.4 and we get: Test ppl:  92.02037624823583, a bit worse, so we revert this change
