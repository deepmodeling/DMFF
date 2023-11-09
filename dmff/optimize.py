import jax
from jax import grad
import jax.numpy as jnp
from typing import Optional
import optax

PeriodicParamsState = optax._src.base.EmptyState


def periodic_move(pmin, pmax):
    def init_fn(params):
        del params
        return PeriodicParamsState()

    def update_fn(updates, state, params):
        if params is None:
            raise ValueError(optax._src.base.NO_PARAMS_MSG)

        updates = jax.tree_map(
            lambda p, u: jnp.where((p + u) < pmin, u + pmax - pmin, u), params, updates
        )
        updates = jax.tree_map(
            lambda p, u: jnp.where((p + u) > pmax, u - pmax + pmin, u), params, updates
        )
        return updates, state

    return optax._src.base.GradientTransformation(init_fn, update_fn)


def genOptimizer(
    optimizer="adam",
    learning_rate=1.0,
    nonzero=True,
    clip=10.0,
    periodic=None,
    transition_steps=1000,
    warmup_steps=0,
    decay_rate=0.99,
    options: dict = {},
):
    if decay_rate == 1.0 and warmup_steps == 0:
        options["learning_rate"] = learning_rate
    # Exponential decay of the learning rate.
    elif warmup_steps == 0:
        scheduler = optax.exponential_decay(
            init_value=learning_rate,
            transition_steps=transition_steps,
            decay_rate=decay_rate,
        )
        options["learning_rate"] = scheduler
    else:
        scheduler = optax.warmup_exponential_decay_schedule(
            init_value=0,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
            transition_steps=transition_steps,
            decay_rate=decay_rate,
        )
        options["learning_rate"] = scheduler

    # Combining gradient transforms using `optax.chain`.
    if optimizer == "sgd":
        chain = [optax.sgd(**options), optax.clip(clip)]
    elif optimizer == "nesterov":
        chain = [optax.sgd(nesterov=True, **options), optax.clip(clip)]
    elif optimizer == "adam":
        chain = [optax.adam(**options), optax.clip(clip)]
    elif optimizer == "adagrad":
        chain = [optax.adagrad(**options), optax.clip(clip)]
    elif optimizer == "adamw":
        chain = [optax.adamw(**options), optax.clip(clip)]
    elif optimizer == "rmsprop":
        chain = [optax.rmsprop(**options), optax.clip(clip)]
    else:
        print(f"Unknown optimizer {optimizer}.")

    if periodic is not None:
        chain.append(periodic_move(periodic[0], periodic[1]))
    elif nonzero:
        chain.append(optax.keep_params_nonnegative())
    gradient_transform = optax.chain(*chain)
    return gradient_transform


def label_iter(parent, ltree, label):
    for key in parent:
        if label:
            newl = f"{label}/{key}"
        else:
            newl = key
        if isinstance(parent[key], dict):
            label_iter(parent[key], ltree, newl)
        else:
            child = ltree
            for k2 in label.split("/"):
                if k2 not in child:
                    child[k2] = {}
                child = child[k2]
            child[key] = newl


def mark_iter(parent, mtree):
    for key in parent:
        if isinstance(parent[key], dict):
            mtree[key] = {}
            mark_iter(parent[key], mtree[key])
        else:
            mtree[key] = False


def label2trans_iter(parent, mtree, ttree):
    for key in parent:
        if isinstance(parent[key], dict):
            label2trans_iter(parent[key], mtree[key], ttree)
        else:
            label = parent[key]
            if label in ttree:
                mtree[key] = True
                # print(label, True)
            else:
                mtree[key] = False
                # print(label, False)
                ttree[label] = optax.set_to_zero()


class MultiTransform:
    def __init__(self, param_tree):
        self.transforms = {}
        self.labels = {}
        self.mask = {}
        label_iter(param_tree, self.labels, "")
        mark_iter(self.labels, self.mask)

    def __getitem__(self, key):
        return self.transforms[key]

    def __setitem__(self, key, val):
        self.transforms[key] = val

    def __delitem__(self, key):
        del self.transforms[key]

    def finalize(self):
        label2trans_iter(self.labels, self.mask, self.transforms)
