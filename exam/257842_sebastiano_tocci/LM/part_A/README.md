**This file is not mandatory**
But if you want, here your can add your comments or anything that you want to share with us
regarding the exercise.

## Here all the different runs and their outcomes are explained:

- Run 1: this is the baseline, where we use the RNN as implemented originally in the lab
        The final outcome was: T"est ppl:  5820.286057093936", which is a very high value. This is probaby due to the learning rate of 0.0001, which is remarkably too small for SDG.

- Run 2: Here we try to overcome the main limitation of the previous run, which was the learning rate. To do so, we use the value provided in [Mikolov et all](https://www.fit.vut.cz/research/group/speech/public/publi/2010/mikolov_interspeech2010_IS100722.pdf), which is 0.1 with SDG. We got: Test ppl:  166.53323616060734. Out of all the subsequent tests with vanilla RNN this was the best model, and for this reason the bin file is preserved inside the ./bin directory, with the name 'training_run_2.pt'

- Run 3: in the previous run we found a good enough learning rate, so now we try to probe around the
        embedding and hidden size. We try to halve them, so they are 150 and 100 respectively. We get Test ppl:  211.22445679466463, so they are too small.
- Run 4: Then we try to go in the other direction: we increase both the embedding and hidden size by +30% compared to the original of 200 and 300. We get: Test ppl:  172.2856682692377, much better than in run 3 but still worse than 2, so we overall did not improve anything.

- Run 5: Always building on the embedding / hidden size tweaking, this time we attempt a 10% increase over the baseline (300, 200 respectively) and see how it goes. We get Test ppl:  168.2143888617081, meaning that essentially what we had before was nearly the best.

- Run 6: Now it's time to move to the LSTS, as detailed in the class instructions. We use the learning rate of 0.1, since it was the setting that yielded the best result with the vanilla RNN. Here we get a wopping PPL of 171.2955283706454.

- Run 7: Here we just increase the learning rate to 0.2 to see what happens: we get an extremely good PPL of 155.80674386671538!

- Run 8: Given the extreme success of increasing the LR, we now set it to 1. We get: Test ppl:  144.05684148483027 in only 32 training epochs, as we quickly run out of patience! This is due to the non adaptive nature of the LR in SGD, which does not decrease as we get further into the trainig.

- Run 9: we do one more test with LR = 0.5, we get Test ppl:  147.98920499842413 after 49 epochs, altough the training PPL got lower than before, 154.529563 vs 155.243145 in run 8. Somewhat surprisingly, in run 8 we got a lower ppl: the reason is most likely that having trained for less epochs, the model was less overfitted compared to this variant.

- Run 10: we now move to the dropped out lstm variant! We keep the same LR as before (0.5) and same emb and hidden sies as before (300, 200). We got a VERY GOOD Test ppl:  127.18228855805629. The dropout technique is working properly.

- Run 11; now it's time to indrodue adamw. We start by using all the default values for the hyper parameters (i.e. lr=0.001). We get: Test ppl:  123.89874121472921 after only 16 epochs, because it terminated.

- Run 12: now we aim to make it train for a bit longer, so we halve lr, which becomes 0.0005 respectively. We got it to traing for 32 epochs and got Test ppl:  121.64160275463718, an improvement!

- Run 13: last attempt with learning rate fine tuning, we will now run on an even smaller lr=0.0001, Test ppl:  127.34688861042393, maybe it was too much, we will then go back and use the parameters from run 12 (lr=0.0005)

- Run 14: we now try a smaller batch size (32) to go in a more stochastic direction. The training terminated early (22 epochs) and we got Test ppl:  122.24647877648398, slightly worse than with batch size = 64. We discard the thing then

- Run 15: now we play around with the momentum, in an effort to increase stochasticity. We set the betas = (0.8, 0.99) and we get: Test ppl:  139.15363928387168, huge downgrade, reverting the change

- Run 16: lastly, we try to tweak the weight_decay: since the training has been aborting early all this time, we try to use a smaller weight decay (weight_decay=0.005), so that the weight won't become 0 too soon. We get: Test ppl:  122.9142327571903, which is a touch worse than what we got in run 12, which at this point is our SOTA.

- Run 17: Since lowering the weight decay didn't really work, we try increasing it, namely double the default (weight_decay=0.02), and we get: Test ppl:  119.6787176317079, which is the BEST so far!! So we try to double again!

- Run 18: Given the success of increasing the weight decay, we set it to weight_decay=0.05, and we get Test ppl:  111.81021725514572 in 42 epochs, a VERY good result! let's try to increase it even more

- RUn 19: We set weight_decay=0.1, and after 53 epochs we get: Test ppl:  105.17273540378693, whic is STILL better than before. We increase it even more

- Run 20: We set weight_decay=0.15 and we get: Test ppl:  103.29959758764394, which is still better than before. We stop here and keep this configuration.

- Run 21: We run a last test to see if increasing b_2 yields any positive result, we set then beta=(0.9, 0.9999), we get Test ppl:  105.80228429010073, so it's not good, back to beta=(0.9, 0.999)

So the final optimal hyperparameters we found are: hid_size = 200, emb_size = 300, lr = 0.0005, weight_decay=0.15, betas=(0.9, 0.999)
