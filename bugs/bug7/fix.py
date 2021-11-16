import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

joint_model = tfd.JointDistributionSequential([
    tfd.Normal(loc=0., scale=1., name='z_0'),
    tfd.Independent(tfd.HalfCauchy(loc=tf.zeros([3]), scale=2., name='lambda_k'), reinterpreted_batch_ndims=1),
    lambda lambda_k, z_0: tfd.MultivariateNormalDiag( # z_k ~ MVN(z_0, lambda_k)
        loc=z_0[...,tf.newaxis],
        scale_diag=lambda_k,
        name='z_k'),
])

print(joint_model)

print(joint_model.log_prob(joint_model.sample(4)))
