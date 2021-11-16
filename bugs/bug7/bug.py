import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions 

joint_model = tfd.JointDistributionSequential([
    tfd.Normal(loc=0., scale=1., name='z_0'),       
    tfd.HalfCauchy(loc=tf.zeros([3]), scale=2., name='lambda_k'),
    lambda lambda_k, z_0: tfd.MultivariateNormalDiag( # z_k ~ MVN(z_0, lambda_k)
        loc=z_0[...,tf.newaxis],
        scale_diag=lambda_k,
        name='z_k'),
])

# These work
joint_model.sample()
joint_model.sample(4)
joint_model.log_prob(joint_model.sample())

# This breaks 
joint_model.log_prob(joint_model.sample(4))
