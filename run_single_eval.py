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
parser.add_argument('--bias', type=float, help='bias parameter.')
parser.add_argument('--linear', type=float, help='linear parameter.')
parser.add_argument('--quad', type=float, help='quadratic parameter.')
parser.add_argument('--cubic', type=float, help='cubic parameter.')

parser.add_argument('--wandb_project', default='wandb-demo-2', type=str)
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


parameter_args = {
	'bias': config_raw['bias'],
	'linear': config_raw['linear'],
	'quad': config_raw['quad'],
	'cubic': config_raw['cubic'],
}

# Args for the general execution.
environment_args = {
	'n_points': x_points.shape[0],
	'git_commit': git_commit,
	'git_branch': git_branch,
	'git_is_dirty': git_is_dirty,
}

# Merge the args.
wnb_args = deepcopy(environment_args)
wnb_args.update(deepcopy(parameter_args))

# Make the WandB object.
wnb = wandb.init(project=config_raw['wandb_project'], entity=config_raw['wandb_entity'], config=wnb_args)


# --------------------------------------------------------------------------------------------------------------------------------------- DO LEARNING.


def loss(params):
	tmp_model = foo(**params)
	loss_vmapped = jax.vmap(tmp_model.loss, in_axes=(0, 0))
	loss = loss_vmapped(x_points, y_points)
	return loss.mean()


# Set up the initial model.
test_model = foo(**parameter_args)


# Save a figure for shits and giggles.
plt.figure(figsize=(6, 6))
plt.scatter(x_points, y_points, c='tab:blue', label='Observed points.')
plt.scatter(x_points, test_model.predict(x_points), c='tab:orange', label='Predicted points.')
plt.grid(True)
plt.ylabel('$2 x^3 + 3 x^2 + 4x + 5$')
plt.xlabel('$x$')
plt.savefig('./predictions_final.png', dpi=300)
wnb.log({'predictions_final': wandb.Image("predictions_final.png")})


# Do some final evaluation and stuff...
final_loss = loss(parameter_args)
wnb.summary['losses/final_loss'] = final_loss


# Save the predictions by logging directly.
yp = foo(**parameter_args).predict(x_points)
with open('./predictions_raw.p', 'wb') as f:
	pickle.dump({'x': x_points,
				 'y': y_points,
				 'yp': yp}, f)
wnb.save('./predictions_raw.p')


# Close up.
# This sometimes confuses WandB in notebooks and isn't actually required.
wnb.finish()

