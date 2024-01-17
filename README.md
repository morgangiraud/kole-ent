 # Kozachenko-Leonenko Entropy Estimator

Hey there! 👋 
This little project of mine started off from a cool [tweet by @adad8m](https://twitter.com/adad8m/status/1745108054306381871), which was actually inspired by another [tweet from @gabrielpeyre](https://twitter.com/gabrielpeyre/status/1744962274018894292). 

These tweets showed off how the Kozachenko-Leonenko entropy estimator can visualize particle gradient flow, and I wanted to check it out for myself!

## What's This All About?
I wanted to see how different optimizers, like the Raw Gradient Descent and the Adam optimizer, would play out in this setup. 

The visual differences are pretty neat!

Also, it gave me a great excuse to mess around with Triton and CUDA kernels.

## Cool Features
- We've got the Kozachenko-Leonenko entropy estimator.
- Thanks to it, we've got a KL estimator.
- You can see particles moving with different gradient flows (`outputs` folder).
- Compare how Raw Gradient Descent stacks up against Adam.

## Wanna Try It Out?

You can try the copy and try google collab [here](https://colab.research.google.com/drive/16W3SE6hGOJUAO3N9TCVp0cSz5Spg10hD?usp=sharing)
Or for OSX (and you will need conda for this one): `make install` and then just run the different scripts

<!-- ## Shoutouts -->
<!-- A bit early for this one -->
