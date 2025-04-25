
./getData.sh
python main.py --cuda --model Transformer --batch_size=100 --emsize=128 --nhid=1024 --lr=20 --nhead=4 --epochs=60 --save=waf-model.pt

