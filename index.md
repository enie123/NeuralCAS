## Towards a Neural Computer Algebra System: Solving Differential Equations through Neural Programming

	
### Introduction 

Computer algebra systems have been around since the mid-20th century, and they have been essential to scientists in many fields for working with symbolic expressions. Differential equations, in particular, are used throughout physics in areas including thermodynamics, fluid dynamics, and quantum mechanics. Physicists currently use computer algebra systems such as Sympy for solving these differential equations. However, one of the issues with these approaches is that the computer algebra systems are often hard-coded to handle different classes of differential equations. As a result, there is a limit to the differential equations that these systems can solve. Another issue with computer algebra systems is the amount of time and computational resources required to solve these differential equations which can be large. An approach to tackling these problems is through the use of deep neural networks. This idea is not new in the field of deep learning and dates back to the 1990s. However, all approaches to this problem thus far have relied upon using numerical evaluations of the differential equation to gather a training dataset. This is then extrapolated by a neural network to a general solution. For these approaches, the neural network trained on a given differential equation does not generalize to others. Hence, this is impractical as an alternative to computer algebra systems for solving differential equations.

An alternative to these approaches is proposed by my advisor Forough Arabshahi which is to symbolically manipulate these differential equations through the use of TreeLSTM networks. Forough Arabshahi is currently a postdoctoral associate in the machine learning department at Carnegie Mellon University and her research focuses on neural programming, deep learning, probabilistic graphical models, and spectral learning. These networks are able to train on the induced tree-structured topology of mathematical equations. A general mathematical equation can be broken up into individual nodes within a tree by splitting the equation on each of the binary operators and dividing the two operands into two sides of the tree. Preliminary research in this direction has shown that this approach is promising and the TreeLSTM networks can produce accurate results with high accuracy for verifying the solutions to differential equations. Assuming that research in this direction is successful, these networks can provide an alternative approach that is more powerful than traditional computer algebra systems. Although neural networks cannot achieve 100% accuracy, they can produce results more efficiently and for a broader class of differential equations. For example, current computer algebra systems can only solve a small subset of partial differential equations, so this is one area that we will be exploring. 

Determining solutions to the differential equations can be divided into two primary steps, which are producing a set of candidate solutions and choosing the correct solution from the set of candidate solutions. The first step has been studied in the context of program synthesis. The paper Neural-Guided Deductive Search for Real-Time Program Synthesis from Examples by Kalyan et. al proposes an approach for combining symbolic logic techniques and deep learning for solving the problem of producing programs from input and output examples. This is one of several papers on this topic. In the preliminary research in Arabshahi et. al, they focused on the second step of this process, so it is unclear as to how well current methods can handle the problem of determining a set of candidate solutions. There does not appear to be an obvious choice for the set of candidate solutions for an arbitrary ordinary differential equation. In order to understand this problem of solving differential equations, I will also need to delve more into the computer algebra literature on this topic.  There are a number of papers and surveys of this field which I am beginning to read. Another challenge we face is determining how well these methods extend to partial differential equations (PDE) which are more complex than ordinary differential equations (ODE).  For our project goals, our benchmarks will be the accuracy of the solutions and the percentage of ODEs and PDEs that our system can solve. Based on the aforementioned two-step process, the framework will likely consist of two independent neural network components. 

### Project Goals

Our 100% goal is to determine a system for finding the candidate solutions for ODEs and to extend these methods to PDEs. For the candidate solutions, we can begin by implementing existing methods from the area of program synthesis which will serve as a baseline. We should then be able to package the two components together into a neural framework for solving differential equations.

Our 75% goal is similar to the previous goal. It may turn out that finding the candidate solutions to the differential equations is a difficult problem that takes a lot of time. In this case, we will focus on tackling this problem for ODEs. 

Our 125% goal will include all of the items within the 100% goal. From there we can branch out and explore other potential approaches to this problem of solving differential equations. We can then compare the different methods to see if there is an approach that can serve as a practical alternative to current computer algebra systems for this application. Since scientists need correctness guarantees, we would need to find a way of combining these deep learning-based approaches with some sort of deterministic verifier that can provide feedback on the results obtained from our system. 


### Project Milestones

Our project milestones are outlined below:

Last day of the semester - For this semester I will focus on reading through the background literature so that I have a solid foundation for the work next semester.  In particular, I hope to understand the mathematics behind how current computer systems solve differential equations. Current theory in this area uses differential ideal theory, differential galois theory, lie algebras among other topics. I do not quite have the mathematical background nor time to dedicate myself to learning these topics so I will focus on gaining an intuition for how these topics apply to solving differential equations. I would also like to begin experimenting with different methods of determining candidate solutions for differential equation. 

 February 1st - I will have finished implementing and training two different methods of determining candidate solutions. Training deep neural networks can be extremely time consuming as a result of the training time and also hyperparameter optimization. 
 
February 15th - I will have thought about new methods for determining candidate solutions. I will implement a method that is mostly novel and finish training it by this time. 

March 1st - I will have finished combining the two components together into a framework for solving ordinary equations. I will also have finished writing documentation, so that the framework is easy to use for other users. 

March 22nd - By this date, I will have thought about the ways of extending these methods from ordinary differential equations to partial differential equations. I will have begun experimenting with the different proposed solutions.

April 5th - I will have finished implementing a verifier for partial differential equations. By verifier, I am referring to a neural network that is able to check the correctness of a solution with high accuracy. 

April 19th - I will have finished implementing a method for determining candidate solutions for partial differential equations. This might involve extending the methods for ordinary differential equations or coming up with an entirely new approach. 

May 3nd - I will have finished combining the two components for PDEs to obtain an alternative method for solving them. Also, I will have finished putting together documentation for the framework so that other people can also use the framework to solve ordinary and partial differential equations. 

### Resources Needed

For this project, we will use MXNet and potentially Tensorflow for implementing the deep learning components such as the TreeLSTM networks. These are open source and so they are available online. In terms of hardware and machines, I will need access to GPUs to shorten the training time. I can either use GPUs through Amazon web services or physical GPUs. My advisor has access to GPUs which I should be able to use. Also, Amazon usually provides credits for students to use their services for educational purposes and so that would be an alternative solution. 


## Project Milestone 1

For this semester, everything has gone smoothly so far, and there are no major changes in the goal of the project since the project proposal. I have began looking into and experimenting with neural program synthesis methods which will be an important aspect of solving differential equations. A challenge is that some of the code for recent papers is not available online. For example, one of the papers contains research completed at Microsoft research and so the framework is proprietary. However, there are also I had the chance to look into the mathematics behind the solution methods to ordinary differential equations that are used by computer algebra systems. There is an overview for these methods provided in this survey on the field, “Computer Algebra and Differential Equations — An Overview” by Werner M. Seiler. I will provide a brief summary of some of the primary methods. The classes of methods outlined are as follows: local analysis, symmetry analysis, completion, differential ideal theory, differential Galois theory, dynamical systems, and numerical analysis. 

Local analysis is a class of methods for generating approximate solutions to the differential equation in cases where the exact solution cannot be determined. An example of a method belonging to this class is determining the Taylor series at a given point. Symmetry analysis as the name suggests relates to the study of symmetries of the group for different differential equations. If the group is solvable then the equation can be reduced to one of lower order. Differential Galois theory is the differential analog to Galois theory that allows for the determination of Liouvillian function solutions. However, determining the differential Galois group for equations is difficult. This overview reinforces that computer algebra systems must be programmed to handle many different cases. 

For the program synthesis portion, the papers that I’ve read through are the following:  Learning to discover efficient mathematical identities (Zaremba et al., 2014), Neural-guided deductive search for real-time program synthesis from examples (Vijayakumar et al., 2018), Neural program search: Solving programming tasks from description and examples (Polosukhin et al., 2018), The Neuro-Symbolic Concept Learner: Interpreting Scenes, Words, and Sentences From Natural Supervision (Blind review, ICLR 2019). These papers have been informative in terms of learning more about how current state of the art neural program searches work. A central component of most of these is optimizing the way in which we search the program space for the correct programs. The programs are specified using domain-specific languages (DSL) in all of the papers. From these papers, I think I have a good idea of how I will approach tackling the problem of generating candidate solutions for the differential equation solver. The challenge of generating candidate solutions can be broken down into subproblems which are first creating a DSL for this problem and then creating a set of input-output pairs for the programming task.

Based on the mathematics behind solving differential equations currently, solving differential equations may turn out to more challenging than expected. In this case, I am thinking that it may be possible to utilize a hybrid approach using more traditional computer algebra techniques combined with neural programming in order to achieve state of the art results. 

There haven’t been any major surprises so far for the project. My milestone goals for the Spring remain the same. I also have all of the resources that I will need for my 15-400 project. 


## Project Milestone 2


Since the end of the last semester there have been some major changes in the goals. It appears that we will probably not be able to get to solving partial differential equations this semester since the challenge of solving just ordinary differential equations is still fairly significant. 
In the last semester, I read about relevant literature; however, I didn’t really have a chance to read through the code base for previous projects that my mentor worked on in the past that are relevant for this project. 
Since the last meeting, I have mostly worked on reading more papers that are relevant to this research which included several papers related to combining modules in dynamic networks. 
I did not meet my milestone for this week because it turns out that determining candidate solutions turns out to be more challenging than I had expected. 
Surprises: Have there been any major surprises in your project since your last bi-weekly status
meeting? If they were bad surprises, have you managed to work around them?
Looking Ahead: What do you plan to focus on and accomplish over the next two weeks?
In the next two weeks, I hope to sort out all of the logistics such as computational resources so that I can focus on experimenting different solutions to the problem. 
As mentioned earlier, I will probably take more time to think about and implement different solutions to this problem instead of trying to tackle partial differential equations as well. 
I am trying to use the PSC in order to run some of the experiments. However, I think there is a memory cap which prevents me from training networks which are memory intensive. 


## Project Milestone 3

There have not been any major changes in goals or implementation of the project since last week. Since the last biweekly meeting, I’ve been reading more papers related to this problem. Another new problem that we identified is augmenting the TreeLSTM networks with memory. For the verification portion of the problem, one of the issues is that the networks only work up to a certain depth so augmenting memory could be a way of enabling these to work up to a greater depth. Some possible solutions for augmenting the memory for these systems is to apply a neural stack or queue. I read the papers that introduced these new neural structures. 

Since I was behind on the last milestone, I am also behind the original milestone for this week. So, the update milestone for next week is to sort out the situation with the computational resources and to also finish implementing an initial version of the solution to this problem based on the paper mentioned above. So, the general idea for the solution based on the paper above is that we can generate a set of differential equations and solutions using sympy the symbolic mathematics library. We can generate the set of differential equations following the tree structure that we are using. Each node is an operation in the equations and the two sides of the operation form the branches to the children of the node. This generated set serves as the dataset for which we can apply the results of the paper. The general idea of the paper is that we can generate complex structured outputs by basing the output on previous outputs that are similar and then modifying them. These two tasks are called retrieve and edit respectively. In order to tackle the problem of generating candidate solutions, we first retrieve differential equations that are similar to the one we are attempting to solve. One of the problems to tackle is figuring out how to define this similarity between different differential equations. Using the tree structure, we can potentially embed the trees as vector using TreeLSTMs, and we can compare the vectors in order to determine the similarity. A potential roadblock is whether similar differential equations have similar solutions. Testing some basic examples it appears that it is true for the most part. Once we have a way of generating candidate solutions, we can also pipeline this first part with a network for verifying the correct solutions to obtain an end to end system for solving differential equations (ones that can be solved through Laplace transforms). 

I tried using PSC in order to generate equations which we will use for training our networks as described above. However, everytime I run something on PSC the program gets killed before it finishes running. My advisor is also working on getting me access to a GPU cluster, so this situation will hopefully be resolved by the next meeting. 
Over the next two weeks, I will implement a version of the retrieve-edit framework for this problem and begin testing the viability of such as solution for solving these differential equations. I should be acquiring access to GPU clusters by the end of this week. As mentioned before the PSC does not appear to work. I am not sure if I need special access in order to use them properly. 

## Project Milestone 4

There have not been any more change in goals. We have decided that we will begin by focusing on augmenting the TreeLSTM networks with memory. In order to augment the TreeLSTM networks with memory, I have been working with my advisor to think about how the equations will change will shifting from the original implementation of TreeLSTM networks to the new implementation. 

The current idea that we are now working on as mentioned above is using memory-augmented TreeLSTM networks in order to increase the depth at which the networks are able to verify the solutions to the differential equations. Currently, maximum depth for the networks without memory is around 7, so we hope to increase this maximum depth through the memory-augmentation. For the next two weeks, I hope to work out the details of the implementation for the new network and begin experimentation.  

The issue with computational resources has also been resolved. I have obtained access to a GPU cluster, so I will be able to train networks on there.  


## Project Milestone 5

There have not been any major changes in the goals since the last meeting. 

Since the last meeting, I have been working on the implementation for the new network. I am close to done with the implementation of the memory-augmented version of the TreeLSTM network. The equations for the new network have been worked out already, so what remains is just a few implementation details. The details for the memory-augmented version can be found in the github repository. 

I would have liked to made slightly more progress for this milestone. However, during spring break, I lost my laptop which slowed down my progress. 

There have not been any major surprises since the last meeting besides losing my laptop, which shouldn’t really affect my progress too much once I get back up to speed. 

For the next two weeks, I will be working on finishing up the implementation of the memory-augmented TreeLSTM network and running experiments to see how well the new network performs. Based on the results, it is likely that some adjustments will have to be made to the implementation details in order for the new network to perform well. I am aiming to reach the point where there is significant improvement in 

At this point I should have almost all of the resources that I need for this project. 





