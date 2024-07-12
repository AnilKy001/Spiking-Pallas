import jax
import jax.numpy as jnp
import equinox as eqx
import os
from jax.experimental import pallas as pl
import optax
from typing import NamedTuple

# For NMNIST dataset:
import tonic
import torch
import tqdm
import time

from typing import Union, List

BATCH_SIZE = 32
N_TIMESTEPS = 8
N_EPOCHS = 5

key = jax.random.PRNGKey(1234)

sensor_size = tonic.datasets.NMNIST.sensor_size
frame_transform = tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=N_TIMESTEPS)
denoise_transform = tonic.transforms.Denoise(filter_time=10000)

transform = tonic.transforms.Compose([denoise_transform, frame_transform])

train_set = tonic.datasets.NMNIST(save_to="./tutorials/nmnist", train=True, transform=transform)
test_set = tonic.datasets.NMNIST(save_to="./tutorials/nmnist", train=False, transform=transform)

torch.manual_seed(1234)

dataloader_train = torch.utils.data.DataLoader(train_set, batch_size = BATCH_SIZE, shuffle=True, drop_last = True)
dataloader_test = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, drop_last = True)

def get_superspike_surrogate(beta = 10.):
    @jax.custom_jvp
    def superspike_surrogate(x):
        return jnp.heaviside(x, 0)

    @superspike_surrogate.defjvp
    def f_jvp(primals, tangents):
        x, = primals
        x_dot, = tangents
        primal_out = superspike_surrogate(x)
        tangent_out = 1. / (beta * jnp.abs(x) + 1.)**2 * x_dot
        return primal_out, tangent_out
    
    return superspike_surrogate

smooth_step = get_superspike_surrogate()

class LIFLayerState(NamedTuple):
    U: jnp.ndarray
    I: jnp.ndarray
    S: jnp.ndarray

class LIFDensePopulation(eqx.Module):
    '''
    Definition of one fully connected layer with leaky integrate-fire neurons.
    '''
    fc_layer: eqx.Module
    out_size: int
    alpha: float
    beta: float

    def __init__(self, in_size: int, out_size: int, alpha: Union[float, int], beta: Union[float, int], use_bias: bool = False, *, key):
        '''
        inputs:
            in_size: Size of the input vector.
            out_size: Size of the output vector.
            alpha: Decay rate for the spikes that contribute to the neuron current.
            beta: Decay rate for the neuron current over time.
            use_bias: Whether to use bias in the fully connected layer.
            key: random number generator key.
        '''
        super().__init__()
        key1, key2, _ = jax.random.split(key, 3)
        self.fc_layer  = eqx.nn.Linear(in_size, out_size, use_bias=use_bias, key=key1)
        self.out_size = out_size
        self.alpha = alpha
        self.beta = beta
   
    def __call__(self, state, Sin_t):

        U = self.alpha*(1-jax.lax.stop_gradient(state.S))*state.U + (1-self.alpha)*(20*state.I)
        I = self.beta*state.I + (1-self.beta)*self.fc_layer(Sin_t)
        S = smooth_step(U-1.)
        new_state = LIFLayerState(U, I, S)
        return new_state, S

    def weight_change(self, weight_change_func):
        object.__setattr__(self, 'fc_layer', jax.tree_util.tree_map(weight_change_func, self.fc_layer))
    
    def init_state(self, batch_size: int):
        '''
        Initialize the states of the neurons for the fully connected layer.
        input:
            batch_size -->  Batch size for the network.
        output:
            Object of LIFLayerState class that contains U, I and S values for the layer. Each state has the shape (batch_size, *out_size).
        '''
        key = jax.random.PRNGKey(123)
        return LIFLayerState(*[jax.random.normal(key, (batch_size, self.out_size)) for _ in range(3)])
    
class LIFNetwork(eqx.Module):
    layers: List[eqx.Module]

    def __init__(self, alpha, beta, *, key):
        super().__init__()
        keys = jax.random.split(key, 6)
        self.layers = [
            LIFDensePopulation(512, 256, alpha, beta, key=keys[0]),
            LIFDensePopulation(256, 128, alpha, beta, key=keys[1]),
            LIFDensePopulation(128, 64, alpha, beta, key=keys[2]),
            LIFDensePopulation(64, 16, alpha, beta, key=keys[3]),
        ]
        
    def __call__(self, initial_state, in_spikes):

        def step_lif_network(state, spikes):
            all_states, all_spikes = [], []
            for layer, state_ilay in zip(self.layers, state):
                spikes = spikes.flatten()
                new_state_ilay, spikes = layer(state_ilay, spikes)
                all_states.append(new_state_ilay)
                all_spikes.append(spikes)
            return all_states, all_spikes
        final_state, out_spikes = jax.lax.scan(step_lif_network, initial_state, in_spikes)
        return final_state, out_spikes

    def init_state(self, batch_size):
        return [layer.init_state(batch_size) for layer in self.layers]
    
def calc_accuracy_batch(model, init_state, in_spikes, labels):
    _, spike_out = jax.vmap(model, in_axes=(0, 0))(init_state, in_spikes)
    sum_spikes_last = spike_out[-1].sum(axis=1)
    pred = sum_spikes_last.argmax(axis=-1)
    print('pred: ', pred)
    return (pred == labels).mean()

key1, key = jax.random.split(key)
model = LIFNetwork(0.5, 0.7, key = key1)
params, static = eqx.partition(model, eqx.is_inexact_array)

pbar = tqdm.trange(N_EPOCHS)

start = time.time()

for epoch in pbar: 
    loss = 0
    acc = []
    # Test loop
    
    for Sin, target in dataloader_test:
        initial_state = model.init_state(BATCH_SIZE)
        Sin = jnp.asarray(Sin)
        target = jnp.asarray(target)
        Sin = jax.lax.reshape(Sin, (*Sin.shape[:-3], Sin.shape[-3] * Sin.shape[-2] * Sin.shape[-1]))
        Sin = Sin[:, :, :512]

        acc.append(calc_accuracy_batch(model, initial_state, jnp.asarray(Sin), jnp.asarray(target)))

end = time.time()
print(end - start)