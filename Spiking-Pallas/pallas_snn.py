import jax
import jax.numpy as jnp
import equinox as eqx
import os
from jax.experimental import pallas as pl
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

def lif_kernel(u_ref, i_ref, s_ref, sin_ref, w_ref, uo_ref, io_ref, so_ref):

    alpha = 0.5
    beta = 0.7
    theta = 1.0 
    
    uo_ref[...] = jnp.multiply(alpha * (jnp.ones((32, 256//16), jnp.float32) - s_ref[...]), u_ref[...]) + (1 - alpha) * (20.0 * i_ref[...])
    io_ref[...] = beta * i_ref[...] + (1 - beta) * (sin_ref[...] @ w_ref[...])
    so_ref[...] = smooth_step(uo_ref[...] - jnp.ones((32, 256//16), jnp.float32) * theta)

def lif_kernel_2(u_ref, i_ref, s_ref, sin_ref, w_ref, uo_ref, io_ref, so_ref):

    alpha = 0.5
    beta = 0.7
    theta = 1.0 
    
    uo_ref[...] = jnp.multiply(alpha * (jnp.ones((32, 128//8), jnp.float32) - s_ref[...]), u_ref[...]) + (1 - alpha) * (20.0 * i_ref[...])
    io_ref[...] = beta * i_ref[...] + (1 - beta) * (sin_ref[...] @ w_ref[...])
    so_ref[...] = smooth_step(uo_ref[...] - jnp.ones((32, 128//8), jnp.float32) * theta)

def lif_kernel_3(u_ref, i_ref, s_ref, sin_ref, w_ref, uo_ref, io_ref, so_ref):

    alpha = 0.5
    beta = 0.7
    theta = 1.0 
    
    uo_ref[...] = jnp.multiply(alpha * (jnp.ones((32, 64//4), jnp.float32) - s_ref[...]), u_ref[...]) + (1 - alpha) * (20.0 * i_ref[...])
    io_ref[...] = beta * i_ref[...] + (1 - beta) * (sin_ref[...] @ w_ref[...])
    so_ref[...] = smooth_step(uo_ref[...] - jnp.ones((32, 64//4), jnp.float32) * theta)

def lif_kernel_4(u_ref, i_ref, s_ref, sin_ref, w_ref, uo_ref, io_ref, so_ref):

    alpha = 0.5
    beta = 0.7
    theta = 1.0 
    
    uo_ref[...] = jnp.multiply(alpha * (jnp.ones((32, 16), jnp.float32) - s_ref[...]), u_ref[...]) + (1 - alpha) * (20.0 * i_ref[...])
    io_ref[...] = beta * i_ref[...] + (1 - beta) * (sin_ref[...] @ w_ref[...])
    so_ref[...] = smooth_step(uo_ref[...] - jnp.ones((32, 16), jnp.float32) * theta)

def lif_layer_pass(u, i, s, s_in, w):
    return pl.pallas_call(
        lif_kernel,
        grid=(1, 16),
        in_specs=[
            pl.BlockSpec(lambda i, j: (i, j), (32, 256//16)),
            pl.BlockSpec(lambda i, j: (i, j), (32, 256//16)),
            pl.BlockSpec(lambda i, j: (i, j), (32, 256//16)),
            pl.BlockSpec(lambda i, j: (0, 0), (32, 512)),
            pl.BlockSpec(lambda i, j: (0, j), (512, 256//16))
        ],
        out_specs=[
            pl.BlockSpec(lambda i, j: (0, j), (32, 256//16)),
            pl.BlockSpec(lambda i, j: (0, j), (32, 256//16)),
            pl.BlockSpec(lambda i, j: (0, j), (32, 256//16))
        ],
        out_shape=(jax.ShapeDtypeStruct((32, 256//16), jnp.float32), jax.ShapeDtypeStruct((32, 256//16), jnp.float32), jax.ShapeDtypeStruct((32, 256//16), jnp.float32))
    )(u, i, s, s_in, w)

def lif_layer_pass_2(u, i, s, s_in, w):
    return pl.pallas_call(
        lif_kernel_2,
        grid=(1, 8),
        in_specs=[
            pl.BlockSpec(lambda i, j: (0, j), (32, 128//8)),
            pl.BlockSpec(lambda i, j: (0, j), (32, 128//8)),
            pl.BlockSpec(lambda i, j: (0, j), (32, 128//8)),
            pl.BlockSpec(lambda i, j: (0, 0), (32, 256)),
            pl.BlockSpec(lambda i, j: (0, j), (256, 128//8))
        ],
        out_specs=[
            pl.BlockSpec(lambda i, j: (0, j), (32, 128//8)),
            pl.BlockSpec(lambda i, j: (0, j), (32, 128//8)),
            pl.BlockSpec(lambda i, j: (0, j), (32, 128//8))
        ],
        out_shape=(jax.ShapeDtypeStruct((32, 128//8), jnp.float32), jax.ShapeDtypeStruct((32, 128//8), jnp.float32), jax.ShapeDtypeStruct((32, 128//8), jnp.float32))
    )(u, i, s, s_in, w)

def lif_layer_pass_3(u, i, s, s_in, w):
    return pl.pallas_call(
        lif_kernel_3,
        grid=(1, 4),
        in_specs=[
            pl.BlockSpec(lambda i, j: (0, j), (32, 64//4)),
            pl.BlockSpec(lambda i, j: (0, j), (32, 64//4)),
            pl.BlockSpec(lambda i, j: (0, j), (32, 64//4)),
            pl.BlockSpec(lambda i, j: (0, 0), (32, 128)),
            pl.BlockSpec(lambda i, j: (0, j), (128, 64//4))
        ],
        out_specs=[
            pl.BlockSpec(lambda i, j: (0, j), (32, 64//4)),
            pl.BlockSpec(lambda i, j: (0, j), (32, 64//4)),
            pl.BlockSpec(lambda i, j: (0, j), (32, 64//4))
        ],
        out_shape=(jax.ShapeDtypeStruct((32, 64//4), jnp.float32), jax.ShapeDtypeStruct((32, 64//4), jnp.float32), jax.ShapeDtypeStruct((32, 64//4), jnp.float32))
    )(u, i, s, s_in, w)

def lif_layer_pass_4(u, i, s, s_in, w):
    return pl.pallas_call(
        lif_kernel_4,
        grid=(1, 1),
        in_specs=[
            pl.BlockSpec(lambda i, j: (0, 0), (32, 16)),
            pl.BlockSpec(lambda i, j: (0, 0), (32, 16)),
            pl.BlockSpec(lambda i, j: (0, 0), (32, 16)),
            pl.BlockSpec(lambda i, j: (0, 0), (32, 64)),
            pl.BlockSpec(lambda i, j: (0, 0), (64, 16))
        ],
        out_specs=[
            pl.BlockSpec(lambda i, j: (0, j), (32, 16)),
            pl.BlockSpec(lambda i, j: (0, j), (32, 16)),
            pl.BlockSpec(lambda i, j: (0, j), (32, 16))
        ],
        out_shape=(jax.ShapeDtypeStruct((32, 16), jnp.float32), jax.ShapeDtypeStruct((32, 16), jnp.float32), jax.ShapeDtypeStruct((32, 16), jnp.float32))
    )(u, i, s, s_in, w)

class LIF_pallas():
    def __init__(self):
        key = jax.random.PRNGKey(12)


    def __call__(self, states, weights, s_in):
        # S_in: batch x feature
        new_states = {'u': [], 'i': [], 's': []}
        
        u, i, s = lif_layer_pass(states['u'][0], states['i'][0], states['s'][0], s_in, weights[0])
        new_states['u'].append(u)
        new_states['i'].append(i)
        new_states['s'].append(s)

        u, i, s = lif_layer_pass_2(states['u'][1], states['i'][1], states['s'][1], new_states['s'][-1], weights[1])
        new_states['u'].append(u)
        new_states['i'].append(i)
        new_states['s'].append(s)

        u, i, s = lif_layer_pass_3(states['u'][2], states['i'][2], states['s'][2], new_states['s'][-1], weights[2])
        new_states['u'].append(u)
        new_states['i'].append(i)
        new_states['s'].append(s)

        u, i, s = lif_layer_pass_4(states['u'][3], states['i'][3], states['s'][3], new_states['s'][-1], weights[3])
        new_states['u'].append(u)
        new_states['i'].append(i)
        new_states['s'].append(s)
        return new_states
       
model = LIF_pallas()

keys = jax.random.split(key, 16)

weights = [
    jax.random.normal(keys[0], (512, 256)),
    jax.random.normal(keys[1], (256, 128)),
    jax.random.normal(keys[2], (128, 64)),
    jax.random.normal(keys[3], (64, 16))
]

states = {}
states['u'] = [
    jax.random.normal(keys[4], (32, 256)),
    jax.random.normal(keys[5], (32, 128)),
    jax.random.normal(keys[6], (32, 64)),
    jax.random.normal(keys[7], (32, 16))
]

states['i'] = [
    jax.random.normal(keys[8], (32, 256)),
    jax.random.normal(keys[9], (32, 128)),
    jax.random.normal(keys[10], (32, 64)),
    jax.random.normal(keys[11], (32, 16))
]

states['s'] = [
    jax.random.normal(keys[12], (32, 256)),
    jax.random.normal(keys[13], (32, 128)),
    jax.random.normal(keys[14], (32, 64)),
    jax.random.normal(keys[15], (32, 16))
]

def calc_accuracy_batch_pallas(model, states, weights, in_spikes, labels):
    state_out = jax.vmap(model, in_axes=(None, None, 1))(states, weights, in_spikes)
    sum_spikes_last = state_out['s'][-1].sum(axis=0)
    pred = sum_spikes_last.argmax(axis=-1)
    print('pred: ', pred)
    return (pred == labels).mean()

pbar = tqdm.trange(N_EPOCHS)

start = time.time()

for epoch in pbar: 
    loss = 0
    acc = []
    # Test loop
    
    for Sin, target in dataloader_test:
        Sin = jnp.asarray(Sin)
        target = jnp.asarray(target, dtype=jnp.float32)
        Sin = jax.lax.reshape(Sin, (*Sin.shape[:-3], Sin.shape[-3] * Sin.shape[-2] * Sin.shape[-1])).astype(jnp.float32)
        Sin = Sin[:, :, :512]
        
        acc.append(calc_accuracy_batch_pallas(model, states, weights, jnp.asarray(Sin), jnp.asarray(target)))

end = time.time()
print(end - start)