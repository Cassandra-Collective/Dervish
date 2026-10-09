# E6 Replication of the best E5 arm (post-hoc)

Command:

```
E6_ARGS="--optimizer dervish --dervish-lr 0.0125 --dervish-scale-lr 0.01 --dervish-cap 1.0" E6_LABEL=dervish_tuned ./run_matrices.sh e6
```

| Config | Seeds | Trainable params | Mean val NLL | Min val NLL | Max val NLL | Mean bits/char | Peak att logit (mean) | Final spectral max (mean) | Final stable rank (mean) | Final out logit max (mean) | Diverged | Mean train s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dervish_tuned | 3 | 802596 | 1.592873 | 1.585442 | 1.603042 | 2.298031 | 16.34 | 1.000 | 59.44 | 11.61 | 0 | 299.1 |

Raw rows:

| Config | Seed | val NLL | bits/char | peak att logit | final att logit | final spectral max | final stable rank | final out logit max | diverged at | train s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dervish_tuned | 7 | 1.590136 | 2.294082 | 15.66 | 15.25 | 1.000 | 60.57 | 11.63 | · | 301.8 |
| dervish_tuned | 11 | 1.603042 | 2.312700 | 16.09 | 16.06 | 1.000 | 58.55 | 11.49 | · | 300.7 |
| dervish_tuned | 19 | 1.585442 | 2.287310 | 17.26 | 17.26 | 1.000 | 59.20 | 11.70 | · | 294.7 |

Evaluation curves (means over seeds):

`dervish_tuned`

| step | val NLL | att logit max | out logit max | spectral max | stable rank |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3.5876 | 0.29 | 0.94 | 0.499 | 85.08 |
| 100 | 2.1871 | 11.38 | 6.74 | 1.000 | 41.83 |
| 200 | 1.8693 | 13.72 | 8.48 | 1.000 | 42.84 |
| 300 | 1.7547 | 14.52 | 9.43 | 1.000 | 45.83 |
| 400 | 1.6922 | 14.99 | 9.77 | 1.000 | 49.49 |
| 500 | 1.6612 | 15.64 | 10.42 | 1.000 | 52.04 |
| 600 | 1.6453 | 15.70 | 10.87 | 1.000 | 54.83 |
| 700 | 1.6281 | 15.71 | 11.27 | 1.000 | 56.43 |
| 800 | 1.6184 | 15.36 | 11.14 | 1.000 | 57.74 |
| 900 | 1.6066 | 15.86 | 10.90 | 1.000 | 58.60 |
| 1000 | 1.5929 | 16.19 | 11.61 | 1.000 | 59.44 |

