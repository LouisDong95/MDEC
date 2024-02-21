cd DEC
python DEC.py --dataset fashion_mnist --save_dir ../results/dec/fashion_mnist

cd ..

cd IDEC
python IDEC.py --dataset fashion_mnist --ae_weights ../results/dec/fashion_mnist/1/ae_weights.h5 --save_dir ../results/idec/fashion_mnist

cd ..

cd DCEC
python DCEC.py --dataset fashion_mnist --save_dir ../results/dcec/fashion_mnist

cd ..

cd DVAE
python DVAE.py --dataset fashion_mnist --save_dir ../results/dvae/fashion_mnist

cd ..

cd MDECfusion
python MDECfusion.py --dataset fashion_mnist --save_dir ../results/mdecfusion/fashion_mnist


