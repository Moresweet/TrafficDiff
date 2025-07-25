import argparse

import torch

# from trainer import train_led_trajectory_augment_input as led
# from trainer import train_led_trajectory_augment_input_hltp_data as led
from trainer import fut_train_led_trajectory_augment_input_hltp_data as led

import os


def parse_config():
	parser = argparse.ArgumentParser()
	parser.add_argument("--cuda", default=True)
	# parser.add_argument("--learning_rate", type=int, default=0.002)
	parser.add_argument("--max_epochs", type=int, default=128)

	parser.add_argument('--cfg', default='led_augment')
	parser.add_argument('--gpu', type=int, default=0, help='Specify which GPU to use.')
	parser.add_argument('--train', type=int, default=1, help='Whether train or evaluate.')
	
	parser.add_argument("--info", type=str, default='', help='Name of the experiment. '
															 'It will be used in file creation.')
	parser.add_argument("--local_rank", type=int, default=-1, help='Name of the experiment. '
															 'It will be used in file creation.')
	parser.add_argument("--pretrain", type=bool, default=False, help='Is train pretrain denoise model?')
	return parser.parse_args()

def main(config):
	t = led.Trainer(config, config.local_rank)
	if config.train==1:
		t.fit()
	else:
		# t.save_data_multi_path()
		t.save_data_multi_path_hltp()
		if config.pretrain is False:
			t.save_data()
		else:
			t.save_data_new()
		t.test_single_model()


if __name__ == "__main__":
	config = parse_config()
	main(config)
