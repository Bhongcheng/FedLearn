# sharding

python main.py --config_path ./config/fedavg.json --dataset_name cifar10 --partition_method sharding --partition_s 2

python main.py --config_path ./config/fedavg.json --dataset_name cifar10 --partition_method sharding --partition_s 5

python main.py --config_path ./config/fedavg.json --dataset_name cifar10 --partition_method sharding --partition_s 10

python main.py --config_path ./config/fedavg.json --dataset_name cifar100 --partition_method sharding --partition_s 10

python main.py --config_path ./config/fedavg.json --dataset_name cifar100 --partition_method sharding --partition_s 50

python main.py --config_path ./config/fedavg.json --dataset_name cifar100 --partition_method sharding --partition_s 100

# lda

python main.py --config_path ./config/fedavg.json --dataset_name cifar10 --partition_method lda --partition_alpha 0.1

python main.py --config_path ./config/fedavg.json --dataset_name cifar10 --partition_method lda --partition_alpha 0.3

python main.py --config_path ./config/fedavg.json --dataset_name cifar10 --partition_method lda --partition_alpha 0.5

python main.py --config_path ./config/fedavg.json --dataset_name cifar10 --partition_method lda --partition_alpha 1.0

python main.py --config_path ./config/fedavg.json --dataset_name cifar100 --partition_method lda --partition_alpha 0.1

python main.py --config_path ./config/fedavg.json --dataset_name cifar100 --partition_method lda --partition_alpha 0.3

python main.py --config_path ./config/fedavg.json --dataset_name cifar100 --partition_method lda --partition_alpha 0.5

python main.py --config_path ./config/fedavg.json --dataset_name cifar100 --partition_method lda --partition_alpha 1.0