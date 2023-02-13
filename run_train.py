import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import wandb
import git
import optax
import os
import numpy as onp
import pickle
import argparse
from copy import deepcopy


# ------------------------------------------------------------------------------------------------------------------------------------- GENERATE DATA.


class foo:
	bias: float = 0.0
	linear: float = 0.0
	quad: float = 0.0
	cubic: float = 0.0

	def __init__(self, bias, linear, quad, cubic):
		self.bias = bias
		self.linear = linear
		self.quad = quad
		self.cubic = cubic

	def predict(self, x_points):
		return (self.bias +
				self.linear * x_points +
				self.quad * (x_points ** 2) +
				self.cubic * (x_points ** 3))

	def loss(self, x_points, y_points):
		y_pred = self.predict(x_points)
		loss_val = ((y_pred - y_points) ** 2).mean()
		return loss_val


foo_true = foo(5, 4, 3, 2)
x_points = jax.random.normal(jax.random.PRNGKey(0), shape=(100,))
y_points = foo_true.predict(x_points)

plt.figure(figsize=(6, 6))
plt.scatter(x_points, y_points)
plt.grid(True)
plt.ylabel('$2 x^3 + 3 x^2 + 4x + 5$')
plt.xlabel('$x$')


# ---------------------------------------------------------------------------------------------------------------------------------- SET UP ARGUMENTS.


# Set up the experiment.
parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=1e-03, type=float)
parser.add_argument('--optimizer', default='adam', type=str)
parser.add_argument('--opt_steps', default=1_000, type=int)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--wandb_project', default='wandb-demo', type=str)
parser.add_argument('--wandb_entity', default='andrewwarrington', type=str)
config_raw = vars(parser.parse_args())


# Grab some git information.
git_commit = 'NoneFound'
git_branch = 'NoneFound'
git_is_dirty = 'NoneFound'
try:
	repo = git.Repo(search_parent_directories=True)
	git_commit = repo.head.object.hexsha
	git_branch = repo.active_branch
	git_is_dirty = repo.is_dirty()
except:
	print('[WARNING]: Failed to grab git info...')


# Learning args.
learning_args = {
	'lr': config_raw['lr'],
	'optimizer': config_raw['optimizer'],
	'opt_steps': config_raw['opt_steps'],
}

# Args for the general execution.
environment_args = {
	'seed': config_raw['seed'],
	'n_points': x_points.shape[0],
	'git_commit': git_commit,
	'git_branch': git_branch,
	'git_is_dirty': git_is_dirty,
}

# Merge the args.
wnb_args = deepcopy(learning_args)
wnb_args.update(deepcopy(environment_args))

# Write out the notebook name.
os.environ['WANDB_NOTEBOOK_NAME'] = 'wandb-demo.ipynb'

# Make the WandB object.
wnb = wandb.init(project=config_raw['wandb_project'], entity=config_raw['wandb_entity'], config=wnb_args)


# --------------------------------------------------------------------------------------------------------------------------------------- DO LEARNING.


def loss(params):
	tmp_model = foo(**params)
	loss_vmapped = jax.vmap(tmp_model.loss, in_axes=(0, 0))
	loss = loss_vmapped(x_points, y_points)
	return loss.mean()


@jax.jit
def train_step(params, opt_state):
	loss_value, grads = jax.value_and_grad(loss)(params)  # Can do minibatching in here.
	updates, opt_state = optimizer.update(grads, opt_state, params)
	params = optax.apply_updates(params, updates)
	return params, opt_state, loss_value


# Set up the initial model.
key = jax.random.PRNGKey(environment_args['seed'])
key, *subkeys = jax.random.split(key, num=5)
model_initial_args = {
	'bias': jax.random.normal(key=subkeys[0]),
	'linear': jax.random.normal(key=subkeys[1]),
	'quad': jax.random.normal(key=subkeys[2]),
	'cubic': jax.random.normal(key=subkeys[3]),
}
test_model = foo(**model_initial_args)
model_args = deepcopy(model_initial_args)

# Set up the optimizer.
if learning_args['optimizer'] == 'adam':
	optimizer = optax.adam(learning_rate=learning_args['lr'])
elif learning_args['optimizer'] == 'sgd':
	optimizer = optax.sgd(learning_rate=learning_args['lr'])
else:
	raise NotImplementedError()

opt_state = optimizer.init(model_initial_args)

# Main training loop.
for step in range(learning_args['opt_steps']):
	model_args, opt_state, loss_value = train_step(model_args, opt_state)

	if step % 100 == 0:
		# Evaluate the model.
		print(f'Step: {step:> 4d}: loss: {loss_value:> 6.4f}')
		wnb.log({'losses/loss': float(loss_value)}, commit=False, step=step)

		# Save a figure for shits and giggles.
		plt.figure(figsize=(6, 6))
		plt.scatter(x_points, y_points, c='tab:blue', label='Observed points.')
		plt.scatter(x_points, foo(**model_args).predict(x_points), c='tab:orange', label='Predicted points.')
		plt.grid(True)
		plt.ylabel('$2 x^3 + 3 x^2 + 4x + 5$')
		plt.xlabel('$x$')
		plt.savefig('./predictions_final.png', dpi=300)
		wnb.log({'predictions_final': wandb.Image("predictions_final.png")}, step=step)

		# Push everything.
		wnb.log({}, commit=True, step=step)

# Do some final evaluation and stuff...
final_loss = loss(model_args)
wnb.summary['losses/final_loss'] = final_loss

# Save the predictions by logging directly.
yp = foo(**model_args).predict(x_points)
with open('./predictions_raw.p', 'wb') as f:
	pickle.dump({'x': x_points,
				 'y': y_points,
				 'yp': yp}, f)
wnb.save('./predictions_raw.p')

# Push anything leftover.
wnb.log({}, commit=True)

# Close up.
# This sometimes confuses WandB in notebooks and isn't actually required.
wnb.finish()







