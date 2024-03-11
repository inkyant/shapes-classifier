
import os
import equinox as eqx
import pickle
import jax.numpy as jnp
import jax
from matplotlib import pyplot as plt
from flax.training import checkpoints
import optax

data = []
labels = []

with open(os.path.join("pickles", "data.pickle"), "rb") as input_file:
    file_data = pickle.load(input_file)
    data = file_data["data"]
    labels = file_data["labels"]


key = jax.random.PRNGKey(4298)
key, trainkey = jax.random.split(key, 2)

indexes = jnp.arange(len(labels))
indexes = jax.random.permutation(trainkey, indexes, independent=True)
first_random_indexes = indexes[:100]


eqx.nn.r


train_split = int(len(indexes)*0.7)

train_data = data[indexes[:train_split]]
train_labels = labels[indexes[:train_split]]

test_data = data[indexes[train_split:]]
test_labels = labels[indexes[train_split:]]

class CNNModel(eqx.Module):
    
    layers: list[eqx.nn.Conv2d | eqx.nn.Linear | eqx.nn.MaxPool2d]
    num_conv_layers: int

    def __init__(self, conv_layers, linear_layers, key):
        self.layers = []
        self.num_conv_layers = len(conv_layers*2)

        for (kernel_size, pool_window_size) in conv_layers:
            key, subkey = jax.random.split(key)
            self.layers.append(
                eqx.nn.Conv2d(
                    1,
                    1,
                    kernel_size,
                    1,
                    0,
                    use_bias=False,
                    key=subkey
                )
            )
            self.layers.append(
                eqx.nn.MaxPool2d(
                    pool_window_size,
                    2
                )
            )

        for (f_in, f_out) in zip(linear_layers[:-1], linear_layers[1:]):
            key, subkey = jax.random.split(key)
            self.layers.append(
                eqx.nn.Linear(
                    f_in,
                    f_out,
                    use_bias = True,
                    key = subkey
                )
            )

    def __call__(self, x):
        a = x
        # i=0
        for layer in self.layers[:self.num_conv_layers]:
            # i += 1
            a = jax.nn.relu(layer(a))
            # display_images(a, [0], alt_title="Conv"+str(i))

        a = jnp.ravel(a)

        for layer in self.layers[self.num_conv_layers:-1]:
            a = jax.nn.relu(layer(a))

        a = jax.nn.sigmoid(self.layers[-1](a))
        return a




# assume square images
input_size = 70

# kernel size for convolution, then window size for max pool
conv_architecture = [(3, 2), (3, 2), (3, 2)]

lin_input_size = input_size

for (kernel_size, max_pool_window_size) in conv_architecture:
    lin_input_size = lin_input_size - kernel_size - max_pool_window_size + 2

linear_architecture = [lin_input_size**2, 128, 128, 4]

model = CNNModel(conv_architecture, linear_architecture, key)
old_test_data = test_data
test_data = jnp.expand_dims(test_data, 1)
train_data = jnp.expand_dims(train_data, 1)

example = test_data[0, :, :, :]
# display_images(example, [0])



# ONLY FOR LOADING MODELS
# from data_processing import display_images

# model_state = checkpoints.restore_checkpoint(
#     ckpt_dir="/saved_models/",  # Folder with the checkpoints
#     target=model,  # matching object to rebuild state in
#     prefix="CNN_shapes",  # Checkpoint file name prefix
# )
# display_images(old_test_data[first_random_indexes], jax.vmap(model)(test_data[first_random_indexes]))
# quit()




# we've created a model, ensure it can actually run (will just output a random value)
print(model(example))


# now lets write a loss function, for multiclass softmax cross entropy is good
def model_loss(model, x, y):

    pred = jax.vmap(model)(x)
    labels_onehot = jax.nn.one_hot(y, 4)

    loss = optax.softmax_cross_entropy(pred, labels_onehot).mean()
    return loss

# test that loss function works
print(model_loss(model,jnp.array([example]), [0]))

# compute gradients. yay jax
model_loss_grad = eqx.filter_value_and_grad(model_loss)

# use adam optimizer
optimizer = optax.adam(0.001)

#adam needs a state, so we want to initialize one
#we can use the exq.filter function to do this, filtering like before
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))


# now we can do a step of our training function
@eqx.filter_jit
def step(m, opt_s, x, y):
    loss, grad = model_loss_grad(m, x, y)
    update, opt_s = optimizer.update(grad, opt_s, m)
    m = eqx.apply_updates(m, update)
    return m, opt_s, loss


# run training loop
l_history = []
n_epochs = 3
print("="*6 + " TRAINING " + "="*6)
for epoch in range(n_epochs):
    model, opt_state, loss = step(model, opt_state, train_data, train_labels)
    l_history.append(loss)
    if epoch % 10 == 0:
        print(f"Epoch {epoch} has loss {loss}")

plt.plot(l_history)
plt.show()



# code to save models

checkpoints.save_checkpoint(
    ckpt_dir="",  # Folder to save checkpoint in
    target=model,  # What to save. To only save parameters, use model_state.params
    step=n_epochs,  # Training step or other metric to save best model on
    prefix="CNN_shapes",  # Checkpoint file name prefix
    overwrite=True,  # Overwrite existing checkpoint files
)


jnp.unique()