# sharding

python main.py --config_path ./config/fedrod.json --dataset_name cifar10 --partition_method sharding --partition_s 2 --device cuda:1

python main.py --config_path ./config/fedrod.json --dataset_name cifar10 --partition_method sharding --partition_s 5 --device cuda:1

python main.py --config_path ./config/fedrod.json --dataset_name cifar10 --partition_method sharding --partition_s 10 --device cuda:1

python main.py --config_path ./config/fedrod.json --dataset_name cifar100 --partition_method sharding --partition_s 10 --device cuda:1

python main.py --config_path ./config/fedrod.json --dataset_name cifar100 --partition_method sharding --partition_s 50 --device cuda:1

python main.py --config_path ./config/fedrod.json --dataset_name cifar100 --partition_method sharding --partition_s 100 --device cuda:1

# lda

python main.py --config_path ./config/fedrod.json --dataset_name cifar10 --partition_method lda --partition_alpha 0.1 --device cuda:1

python main.py --config_path ./config/fedrod.json --dataset_name cifar10 --partition_method lda --partition_alpha 0.3 --device cuda:1

python main.py --config_path ./config/fedrod.json --dataset_name cifar10 --partition_method lda --partition_alpha 0.5 --device cuda:1

python main.py --config_path ./config/fedrod.json --dataset_name cifar10 --partition_method lda --partition_alpha 1.0 --device cuda:1

python main.py --config_path ./config/fedrod.json --dataset_name cifar100 --partition_method lda --partition_alpha 0.1 --device cuda:1

python main.py --config_path ./config/fedrod.json --dataset_name cifar100 --partition_method lda --partition_alpha 0.3 --device cuda:1

python main.py --config_path ./config/fedrod.json --dataset_name cifar100 --partition_method lda --partition_alpha 0.5 --device cuda:1

python main.py --config_path ./config/fedrod.json --dataset_name cifar100 --partition_method lda --partition_alpha 1.0 --device cuda:1