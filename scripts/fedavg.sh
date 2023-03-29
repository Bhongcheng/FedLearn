# cifar10

python main.py --config_path ./config/fedavg.json --dataset_name cifar10 --partition_method sharding --partition_s 2

python main.py --config_path ./config/fedavg.json --dataset_name cifar10 --partition_method sharding --partition_s 3

python main.py --config_path ./config/fedavg.json --dataset_name cifar10 --partition_method sharding --partition_s 5

python main.py --config_path ./config/fedavg.json --dataset_name cifar10 --partition_method sharding --partition_s 10

python main.py --config_path ./config/fedavg.json --dataset_name cifar10 --partition_method lda --partition_alpha 0.05

python main.py --config_path ./config/fedavg.json --dataset_name cifar10 --partition_method lda --partition_alpha 0.1

python main.py --config_path ./config/fedavg.json --dataset_name cifar10 --partition_method lda --partition_alpha 0.3

python main.py --config_path ./config/fedavg.json --dataset_name cifar10 --partition_method lda --partition_alpha 0.5

# cifar100

python main.py --config_path ./config/fedavg.json --dataset_name cifar100 --partition_method sharding --partition_s 10

python main.py --config_path ./config/fedavg.json --dataset_name cifar100 --partition_method lda --partition_alpha 0.1

# mnist

python main.py --config_path ./config/fedavg.json --dataset_name mnist --partition_method sharding --partition_s 2 --model_name fedavg_mnist

python main.py --config_path ./config/fedavg.json --dataset_name mnist --partition_method lda --partition_alpha 0.1 --model_name fedavg_mnist

# cinic10

python main.py --config_path ./config/fedavg.json --dataset_name cinic10 --partition_method sharding --partition_s 2 --n_clients 200 --sample_ratio 0.05

python main.py --config_path ./config/fedavg.json --dataset_name cinic10 --partition_method lda --partition_alpha 0.1 --n_clients 200 --sample_ratio 0.05