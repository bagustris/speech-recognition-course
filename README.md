# Speech Recognition Course

Material for learning speech recognition, based on Microsoft teaching material on EdX (changed from CNTK to PyTorch). Learning/teaching materials are given in each module/directory. The comprehensive learning materials covers signal processing, acoustic modeling, language modeling, and modern end-to-end approaches

Github Pages:  [https://bagustris.github.io/speech-recognition-course](https://bagustris.github.io/speech-recognition-course)  
Repository: [https://github.com/bagustris/speech-recognition-course](https://github.com/bagustris/speech-recognition-course)  

## Modules
- [Module 1: Introduction to Speech Recognition](./M1_Introduction/)
- [Module 2: Speech Signal Processing](./M2_Speech_Signal_Processing/)
- [Module 3: Acoustic Modeling](./M3_Acoustic_Modeling/)
- [Module 4: Language Modeling](./M4_Language_Modeling/)
- [Module 5: Decoding](./M5_Decoding/)
- [Module 6: End-to-End Models](./M6_End_to_End_Models/)

Convert from markdown to pdf with pandoc in each module:

```bash
pandoc readme.md -o readme.pdf
``` 

Then, you can inspect the generated PDFs.

### References:  
1. https://learning.edx.org/course/course-v1:Microsoft+DEV287x+1T2019a/home   
1. L. Gillick and S. J. Cox, “Some statistical issues in the comparison of speech recognition algorithms,” in ICASSP, IEEE International Conference on Acoustics, Speech and Signal Processing - Proceedings, 1989, vol. 1, pp. 532–535, doi: 10.1109/icassp.1989.266481.  
1. M. Mohri, F. Pereira, and M. Riley, “SPEECH RECOGNITION WITH WEIGHTED FINITE-STATE TRANSDUCERS,” in Springer Handbook on Speech Processing and Speech Communication.  
1. D. S. Pallet, W. M. Fisher, and J. G. Fiscus, “Tools for the analysis of benchmark speech recognition tests,” ICASSP, IEEE Int. Conf. Acoust. Speech Signal Process. - Proc., vol. 1, pp. 97–100, 1990, doi: 10.1109/icassp.1990.115546.  
1. T. Morioka, T. Iwata, T. Hori, and T. Kobayashi, “Multiscale recurrent neural network based language model,” in Proceedings of the Annual Conference of the International Speech Communication Association, INTERSPEECH, 2015, vol. 2015-Janua, pp. 2366–2370.  
1. M. Sundermeyer, R. Schlüter, and H. Ney, “LSTM neural networks for language modeling,” in 13th Annual Conference of the International Speech Communication Association 2012, INTERSPEECH 2012, 2012, vol. 1, pp. 194–197, Accessed: Aug. 08, 2020. [Online]. Available: http://www.isca-speech.org/archive.  
1. Y. Bengio, R. Ducharme, and P. Vincent, “A neural probabilistic language model,” J. Mach. Learn. Res., vol. 3, pp. 1137–1155, 2003.  
1. P. F. Brown, P. V DeSouza, R. L. Mercer, V. J. Della Pietra, and J. C. Lai, “Class-Based n-gram Models of Natural Language,” Comput. Linguist., vol. 18, no. 4, pp. 467–480, 1992.  
1. M. Levit, S. Parthasarathy, S. Chang, A. Stolcke, and B. Dumoulin, “Word-phrase-entity language models: Getting more mileage out of N-grams,” in Proceedings of the Annual Conference of the International Speech Communication Association, INTERSPEECH, 2014, pp. 666–670.  
1. X. Shen, Y. Oualil, C. Greenberg, M. Singh, and D. Klakow, “Estimation of gap between current language models and human performance,” in Proceedings of the Annual Conference of the International Speech Communication Association, INTERSPEECH, 2017, vol. 2017-Augus, pp. 553–557, doi: 10.21437/Interspeech.2017-729.  
1.  A. Stolcke, “Entropy-based Pruning of Backoff Language Models,” pp. 579–588, 2000, doi: 10.3115/1075218.107521. 
