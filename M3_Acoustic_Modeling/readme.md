# M3: Acoustic Modeling

## Table of Contents
- [Introduction](#introduction)
- [Hidden Markov Models](#hidden-markov-models)
- [The Evaluation Problem](#the-evaluation-problem)
- [The Decoding Problem](#the-decoding-problem)
- [The Training Problem](#the-training-problem)
- [Hidden Markov Models for Speech Recognition](#hidden-markov-models-for-speech-recognition)
- [Choice of subword units](#choice-of-subword-units)
- [Deep Neural Network Acoustic Models](#deep-neural-network-acoustic-models)
- [Generate Frame based Sonal Levels](#generate-frame-based-sonal-levels)
- [Training Feedforward Deep Neural Networks](#training-feedforward-deep-neural-networks)
- [Training Recurrent Neural Networks](#training-recurrent-neural-networks)
- [Long Short-Term Memory Networks](#long-short-term-memory-networks)
- [Using a Sequence-based Objective Function](#using-a-sequence-based-objective-function)
- [Decoding with Neural Network Acoustic Models](#decoding-with-neural-network-acoustic-models)
- [Lab](#lab)

## Introduction
In this module, we'll talk about the acoustic model used in modern speech recognizers. In most systems today, the acoustic model is a hybrid model with uses deep neural networks to create frame-level predictions and then a hidden Markov model to transform these into a sequential prediction. A hidden Markov model (HMM) is a very well-known method for characterizing a discrete-time (sampled) sequence of events. The basic ideas of HMMs are decades old and have been applied to many fields.

[Rest of content remains unchanged...]