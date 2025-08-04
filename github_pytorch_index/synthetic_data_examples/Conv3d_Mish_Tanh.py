# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


class Conv3d_Mish_Tanh(nn.Module):
	"""
	Model that performs a 3D convolution, applies Mish activation, and then applies Tanh activation.
	"""

	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
		super(Conv3d_Mish_Tanh, self).__init__()
		self.conv = nn.Conv3d(
			in_channels, out_channels, kernel_size, stride=stride, padding=padding
		)

	def forward(self, x):
		"""
		Args:
		    x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

		Returns:
		    torch.Tensor: Output tensor of shape (batch_size, out_channels, D', H', W').
		"""
		x = self.conv(x)
		x = torch.nn.functional.mish(x)
		x = torch.tanh(x)
		return x


batch_size = 16
in_channels = 3
out_channels = 16
D, H, W = 16, 32, 32
kernel_size = 3


def get_inputs():
	return [torch.randn(batch_size, in_channels, D, H, W)]


def get_init_inputs():
	return [[in_channels, out_channels, kernel_size], {}]
