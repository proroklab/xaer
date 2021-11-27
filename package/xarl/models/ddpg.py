from ray.rllib.agents.ddpg.ddpg_tf_model import DDPGTFModel
from xarl.models.adaptive_model_wrapper import get_tf_heads_model, get_heads_input, tf

class TFAdaptiveMultiHeadDDPG(DDPGTFModel):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, actor_hiddens=(256, 256), actor_hidden_activation="relu", critic_hiddens=(256, 256), critic_hidden_activation="relu", twin_q=False, add_layer_norm=False):
        inputs, last_layer = get_tf_heads_model(obs_space)
        super().__init__(obs_space=obs_space, action_space=action_space, num_outputs=last_layer.shape[1], model_config=model_config, name=name, actor_hiddens=actor_hiddens, actor_hidden_activation=actor_hidden_activation, critic_hiddens=critic_hiddens, critic_hidden_activation=critic_hidden_activation, twin_q=twin_q, add_layer_norm=add_layer_norm)
        self.heads_model = tf.keras.Model(inputs, last_layer)
        self.register_variables(self.heads_model.variables)

    def forward(self, input_dict, state, seq_lens):
        model_out = self.heads_model(get_heads_input(input_dict))
        return model_out, state
