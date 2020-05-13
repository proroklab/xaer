import tensorflow.compat.v1 as tf
#https://github.com/tensorflow/tensorflow/issues/27640

class TFLiteModel():
	
	@staticmethod
	def export_from_session(session, input_nodes, output_nodes, tflite_filename):
		print("Converting to tflite...")
		converter = tf.lite.TFLiteConverter.from_session(
			session, 
			input_nodes, 
			output_nodes
		)
		#converter.post_training_quantize = True
		converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS] 
		# converter.inference_type = tf.lite.constants.QUANTIZED_UINT8
		# converter.allow_custom_ops = True
		tflite_model = converter.convert()
		with open(tflite_filename, "wb") as f:
			f.write(tflite_model)
		print("Converted %s." % tflite_filename)
		
	def __init__(self, tflite_filename):
		print("Loading TFLite interpreter for %s..." % tflite_filename)
		self.interpreter = tf.lite.Interpreter(model_path=tflite_filename)
		self.interpreter.allocate_tensors()
		self.input_details = self.interpreter.get_input_details()
		self.output_details = self.interpreter.get_output_details()
		print("input details: %s" % self.input_details)
		print("output details: %s" % self.output_details)
		
	def run(self, input_dict, output_list):
		for input_tensor in self.input_details:
			self.interpreter.set_tensor(input_tensor['index'], input_dict[input_tensor['name']])
		self.interpreter.invoke()
		output_dict = {
			output_tensor['name']: self.interpreter.get_tensor(output_tensor['index'])
			for output_tensor in self.output_details
			if output_tensor['name'] in output_list
		}
		return output_dict