"""
Converts ovf to png. Choice of vector component and layer.

non-standard packages can be installed with:
pip install numpy pandas pillow
"""

import numpy as np
import os, re, multiprocessing, tkinter
import pandas as pd
from tkinter import filedialog, Label, Button, Entry, Checkbutton
from multiprocessing import Pool, Value
from PIL import Image

task_counter = None

class convert_ovf():
	# for vectors: 0, 1, 2. -1 means the last component
	mag_components_to_convert = -1
	# if false, only convert the middle layer, usually False for magnetisation conversion
	convert_all_layers = False
	layers_to_convert = [-1]
	# normalise or not, False for magnetisation conversion
	normalise = False

	def process_main(self):
		self.get_user_params_UI()
		files = filedialog.askopenfilenames(title='Select .ovf files')
		num_files = len(files)

		task_counter = Value('i', 0)
		pool_input_list = []

		# build pool input list
		for ind, pathandfilename in enumerate(files):
			pool_input_list.append((pathandfilename, num_files))

		num_procs = int(multiprocessing.cpu_count()/2)
		if num_procs == 0:
			num_procs = 1
		# process domains with multiprocessing
		with Pool(processes=num_procs, initializer = self.init, initargs = (task_counter, )) as p:
			# read in all these files as one dataset
			p.starmap(self.do_conversion, pool_input_list)

	def get_user_params_UI(self):
		# construct simple UI to get the parameters for conversion

		def button_callback():
			self.mag_components_to_convert = int(mag_component_input.get())

			tmp_layer_input = layer_input.get().replace(',',' ').split()
			self.layers_to_convert = [int(x) for x in tmp_layer_input]
			self.normalise = normalise_input_val.get()
			window.destroy()

		padding = {'ipadx': 5, 'ipady': 5, 'padx': 5, 'pady': 5}
		window = tkinter.Tk()
		window.option_add('*Font', '20')
		window.title('Conversion Parameters')

		Label(window, text='Please enter parameters for conversion, correctly. My maker is too lazy to implement data validation.').grid(column=0, columnspan=2, row=0, **padding)

		Label(window, text='Mag vector component to convert (0, 1, 2, -1. -1: last component):').grid(column=0, row=1, **padding)
		mag_component_input = Entry(window, width=10)
		mag_component_input.grid(column=1, row=1, **padding)
		mag_component_input.insert(tkinter.INSERT, self.mag_components_to_convert)

		Label(window, text='Layer numbers to convert, separated by space and/or comma (0, 1, 2... , -1: middle layer):').grid(column=0, row=2, **padding)
		layer_input = Entry(window, width=10)
		layer_input.grid(column=1, row=2, **padding)
		layer_input.insert(tkinter.INSERT, '2,3')

		Label(window, text='Whether or not to normalise the output. Turn off for magnetisation conversion.:').grid(column=0, row=3, **padding)
		normalise_input_val = tkinter.BooleanVar()
		normalise_input = Checkbutton(window, width=10, onvalue=True, offvalue=False, variable=normalise_input_val)
		normalise_input.grid(column=1, row=3, **padding)
		if self.normalise:
			normalise_input.select()

		Button(window, text='Okay', command=button_callback).grid(column=0, columnspan=2, row=4, **padding)

		window.mainloop()

	def init(self, args):
		# bind global task_counter to local task_counter (of Value)
		# this is needed to avoid error: RuntimeError: Synchronized objects should only be shared between processes through inheritance
		global task_counter
		task_counter = args

	def do_conversion(self, pathandfilename, num_files):

		global task_counter

		x_len, y_len, z_len, x_step, y_step, z_step, data_out = self.import_ovf(path_and_filename=pathandfilename)

		# select the middle layer, only the mz component
		middle_layer_ind = int(np.floor(z_len/2))
		self.layers_to_convert = [middle_layer_ind if x == -1 else x for x in self.layers_to_convert]
		# remove duplicate by converting to dictionary and back
		self.layers_to_convert = list(dict.fromkeys(self.layers_to_convert))

		for layer in self.layers_to_convert:
			data = data_out[layer, :, :, self.mag_components_to_convert]

			data_filename = os.path.splitext(pathandfilename)[0] + '_layer%d.npy'%layer
			
			np.save(data_filename, np.array(data))
			
			"""
			Note: Used to output a tif image from 0 to 256.

			Justin wants the real data in the form of an np array so commented the code below out.
			"""
			# if self.normalise:
			# 	data = ((data - np.min(data)) / (np.max(data) - np.min(data)) * (2**8-1)).astype(np.uint8)
			# else:
			# 	data = ((data+1)/2*(2**8-1)).astype(np.uint8)

			# # create image, flipud to match mumax simulation preview
			# im = Image.fromarray(np.flipud(data), mode='L')
			# # save image
			# image_filename = os.path.splitext(pathandfilename)[0] + '_layer%d.tif'%layer
			# im.save(image_filename)

		# safe way of reporting completion
		with task_counter.get_lock():
			task_counter.value += 1
			# progress
			print(f'Completed {pathandfilename}, which is {task_counter.value} of out {num_files} ({task_counter.value/num_files*100:0.1f}%).')

	def import_ovf(self, path_and_filename=None, header_bytes=1000):
		"""
		Import the data
		mag_direction: 0-2: x,y,z. Direction of mag to output. Negative to output all
		"""

		# First read the header byte and find the the number of xnodes, ynodes, znodes
		with open(path_and_filename) as fin:
			header_data = fin.read(header_bytes)

		num_nodes = re.search('# xnodes: (\+?-?\d+)\s*# ynodes: (\+?-?\d+)\s*# znodes: (\+?-?\d+)', header_data)
		x_len = int(num_nodes.group(1))
		y_len = int(num_nodes.group(2))
		z_len = int(num_nodes.group(3))

		step_size = re.search('# xstepsize: (\+?-?\d+)e(\+?-?\d+)\s*# ystepsize: (\+?-?\d+)e(\+?-?\d+)\s*# zstepsize: (\+?-?\d+)e(\+?-?\d+)', header_data)
		if not step_size is None:
			x_step = int(step_size.group(1))*10**(int(step_size.group(2)))
			y_step = int(step_size.group(3))*10**(int(step_size.group(4)))
			z_step = int(step_size.group(5))*10**(int(step_size.group(6)))
		else:
			x_step = None
			y_step = None
			z_step = None

		# np.loadtxt is very slow for some reason
		# data_in = np.loadtxt(path_and_filename, skiprows=num_comment_lines)
		data_in = pd.read_csv(path_and_filename, header=None, sep=' ', comment='#').to_numpy()
		num_components = len(data_in[0])-1
		# get rid of the last column of nans
		data_in = data_in[:,0:num_components]
		data_reshaped = data_in.reshape((z_len, y_len, x_len, num_components), order='C')

		# select a single layer
		# data_out = data_reshaped[layer,:,:,:]
		data_out = data_reshaped

		# data_x = data_in[:,0]
		# data_y = data_in[:,1]
		# data_z = data_in[:,2]

		# skip reshape for this usage into original shape
		# data_x = data_x.reshape((z_len, y_len, x_len), order='C')
		# data_y = data_y.reshape((z_len, y_len, x_len), order='C')
		# data_z = data_z.reshape((z_len, y_len, x_len), order='C')

		return x_len, y_len, z_len, x_step, y_step, z_step, data_out

# run the main function
if __name__ == '__main__':
	converter = convert_ovf()
	converter.process_main()