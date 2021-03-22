# Linear-Vector-Quantization-in-an-Online-Setting

Incremental Online Learning Using Learning Vector Quantization (LVQ)

Mason Smith<br/>Chinmayi Shivareddy<br/> Abhijith Venkatesh Kumar<br/>Abdul Zindani<br/>Karan Shah<br/>


The focus of this project will be to implement an incremental online learning architecture using Learning Vector Quantization to classify images. Learning Vector Quantization (LVQ) is a family of algorithms for statistical pattern classification, which aims at learning prototypes representing class regions. Training data is used as prototypes in which the object is classified according to the closest distance to the prototype. Using the model developed for the project, different prototype replacement methods will be implemented and compared. These methods include sampling cost replacement and clustering. The model will be developed using python. In order to develop the model, 70% of the data will be used for training, while 30% will be used as test data, but this is subject to adjustment depending on results of implementation.
Incremental online learning for tasks such as image classification and identification is significant due to its many applications in the fields of biomedical data analysis, image recognition, and robotics. These methods also address the issue of limited memory resources and computational power for the systems that such algorithms may run on. Storing enough prototypes to accurately generate a ubiquitous model might be infeasible for a highly dynamic system. Some systems in image processing may also have long-term trends that would change the usefulness of prototypes (i.e. day/night). Replacing the LVQ model prototypes to match relevant object characteristics would be helpful in reducing need for memory while maintaining validity of the finite prototypes.

<br/>This project draws its main inspiration from the following research paper,
- V. Losing, B. Hammer, Heiko Wersing, “Interactive online learning for obstacle classification on a mobile robot”
The project will also look at the following papers for more information, methods and techniques related to clustering, LVQ, and online learning.
- M. Grbovic and S. Vucetic, “Learning vector quantization with adaptive prototype addition and removal,” in IJCNN, 2009.
- J. G. Dias and M. J. Cortinhal, “The skm algorithm: A k-means algorithm for clustering sequential data.” in IBERAMIA, ser. Lecture Notes in Computer Science, vol. 5290. Springer, 2008.
- A. Sato and K. Yamada, “Generalized learning vector quantization.” in NIPS. MIT Press, 1995.
- T. Kohonen, “An Introduction to Neural Computing,” Neural Networks 1, pp. 3-16, 1988.
- David Nova, Pablo A. Estévez, “A Review of Learning Vector Quantization Classifiers”
- Ruxandra Tapu, Bogdan Mocanu, Andrei Bursuc, Titus Zaharia“A Smartphone-Based Obstacle
Detection and Classification System for Assisting Visually Impaired People” ICCV2013 workshop paper.
The expected outcome of this project is the implementation of an online LVQ algorithm to classify images. The algorithms will be used to compare results between different replacement methods, such as sampling cost replacement and clustering. The results of this comparison are expected to be comparable to those found in the paper by Losing, et al. where sampling cost was the more accurate prototype replacement method.
