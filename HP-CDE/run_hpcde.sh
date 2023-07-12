for seed in 2022
do
for epoch in 100
do
    for batch in 16
    do
        for scale in 10
        do
            for llscale in 100
            do
                for lr in 1e-3
                do
                    for data in 'data/data_mimic/'
                    do
                        for d_model in 70
                        do
                            for decay in 1e-5
                            do
                                for d_ncde in 128
                                do
                                    for model in 'hpcdev1'
                                    do
                                        for n_layers in 6
                                        do
                                            for hh_dim in 90
                                            do
                                                python -u Main_HPCDE.py --data $data --seed $seed --model $model --d_model $d_model --n_layers $n_layers --hh_dim $hh_dim --d_ncde $d_ncde --decay $decay --batch_size $batch --scale $scale --llscale $llscale --epoch $epoch --lr $lr  > ./HP-cdev1.csv
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
done

