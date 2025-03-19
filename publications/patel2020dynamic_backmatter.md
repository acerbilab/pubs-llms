# Dynamic allocation of limited memory resources in reinforcement learning - Backmatter

---

#### Page 10

# Broader Impact 

We believe that this work has the potential to lead to a net-positive change in the reinforcement learning community and more broadly in society as a whole. Our work enables researchers to represent the uncertainty in memories due to resource constraints and perform well in the face of such constraints by prioritizing the knowledge that really matters. While our work is preliminary, we believe that furthering this line of work may prove to be highly beneficial in reducing the overall carbon footprint of the artificial intelligence (AI) industry, which has recently come under scrutiny for the jarring energy consumption of several common large AI models that produce up to five times as much $\mathrm{CO}_{2}$ than an average American car does in its lifetime [51, 52].
In terms of ethical aspects, our method is neutral per se. The advancement of energy-efficient algorithms may enable autonomous agents to function for long hours in remote areas, the applications for which could be used for both constructive and destructive things alike, e.g. they may be deployed for rescue missions [53] or weaponized for military applications [54, 55], but this holds true for any RL agent.

## Acknowledgments and Disclosure of Funding

We thank Pablo Tano and Reidar Riveland for useful discussions, and Morio Hamada for providing helpful feedback on a previous version of the manuscript. Nisheet Patel was supported by the Swiss National Foundation (grant number 31003A_165831). Luigi Acerbi was partially supported by the Academy of Finland Flagship programme: Finnish Center for Artificial Intelligence (FCAI). Alexandre Pouget was supported by the Swiss National Foundation (grant numbers 31003A_165831 and 315230_197296).

## References

[1] Leslie Pack Kaelbling, Michael L Littman, and Andrew W Moore. Reinforcement learning: A survey. Journal of artificial intelligence research, 4:237-285, 1996.
[2] Richard S Sutton, Andrew G Barto, et al. Introduction to reinforcement learning, volume 135. MIT press Cambridge, 1998.
[3] Christopher JCH Watkins and Peter Dayan. Q-learning. Machine learning, 8(3-4):279-292, 1992.
[4] Gavin A Rummery and Mahesan Niranjan. On-line Q-learning using connectionist systems, volume 37. University of Cambridge, Department of Engineering Cambridge, UK, 1994.
[5] Craig Denis Hardman, Jasmine Monica Henderson, David Isaac Finkelstein, Malcolm Kenneth Horne, George Paxinos, and Glenda Margaret Halliday. Comparison of the basal ganglia in rats, marmosets, macaques, baboons, and humans: volume and neuronal number for the output, internal relay, and striatal modulating nuclei. Journal of Comparative Neurology, 445(3):238-255, 2002.
[6] Samuel J Gershman, Eric J Horvitz, and Joshua B Tenenbaum. Computational rationality: A converging paradigm for intelligence in brains, minds, and machines. Science, 349(6245):273-278, 2015.
[7] Falk Lieder and Thomas L Griffiths. Resource-rational analysis: understanding human cognition as the optimal use of limited computational resources. Behavioral and Brain Sciences, 43, 2020.
[8] Susanne Still and Doina Precup. An information-theoretic approach to curiosity-driven reinforcement learning. Theory in Biosciences, 131(3):139-148, 2012.
[9] Jonathan Rubin, Ohad Shamir, and Naftali Tishby. Trading value and information in MDPs. In Decision Making with Imperfect Decision Makers, pages 57-74. Springer, 2012.
[10] Jordi Grau-Moya, Felix Leibfried, Tim Genewein, and Daniel A Braun. Planning with informationprocessing constraints and model uncertainty in Markov decision processes. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases, pages 475-491. Springer, 2016.
[11] Daniel Alexander Ortega and Pedro Alejandro Braun. Information, utility and bounded rationality. In International Conference on Artificial General Intelligence, pages 269-274. Springer, 2011.

---

#### Page 11

[12] Michael T Todd, Yael Niv, and Jonathan D Cohen. Learning to use working memory in partially observable environments through dopaminergic reinforcement. In Advances in neural information processing systems, pages 1689-1696, 2009.
[13] Jordan W Suchow and Tom Griffiths. Deciding to remember: Memory maintenance as a Markov decision process. In CogSci, 2016.
[14] Ronald Van den Berg and Wei Ji Ma. A resource-rational theory of set size effects in human visual working memory. ELife, 7:e34963, 2018.
[15] Tom Schaul, John Quan, Ioannis Antonoglou, and David Silver. Prioritized experience replay. arXiv preprint arXiv:1511.05952, 2015.
[16] Marcelo G Mattar and Nathaniel D Daw. Prioritized memory access explains planning and hippocampal replay. Nature neuroscience, 21(11):1609-1617, 2018.
[17] Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. arXiv preprint arXiv:1801.01290, 2018.
[18] Yifan Wu, George Tucker, and Ofir Nachum. The laplacian in rl: Learning representations with efficient approximations. arXiv preprint arXiv:1810.04586, 2018.
[19] Brendan O’Donoghue, Ian Osband, Remi Munos, and Volodymyr Mnih. The uncertainty Bellman equation and exploration. arXiv preprint arXiv:1709.05380, 2017.
[20] Hongzi Mao, Mohammad Alizadeh, Ishai Menache, and Srikanth Kandula. Resource management with deep reinforcement learning. In Proceedings of the 15th ACM Workshop on Hot Topics in Networks, pages 50-56, 2016.
[21] Hao Ye, Geoffrey Ye Li, and Biing-Hwang Fred Juang. Deep reinforcement learning based resource allocation for V2V communications. IEEE Transactions on Vehicular Technology, 68(4):3163-3173, 2019.
[22] Richard S Sutton and Andrew G Barto. Reinforcement learning: An introduction 2nd ed, 2018.
[23] Andrea Hasenstaub, Stephani Otte, Edward Callaway, and Terrence J Sejnowski. Metabolic cost as a unifying principle governing neuronal biophysics. Proceedings of the National Academy of Sciences, 107 (27):12329-12334, 2010.
[24] Leo P Sugrue, Greg S Corrado, and William T Newsome. Matching behavior and the representation of value in the parietal cortex. science, 304(5678):1782-1787, 2004.
[25] Kazuyuki Samejima, Yasumasa Ueda, Kenji Doya, and Minoru Kimura. Representation of action-specific reward values in the striatum. Science, 310(5752):1337-1340, 2005.
[26] Matthew R Roesch, Donna J Calu, and Geoffrey Schoenbaum. Dopamine neurons encode the better option in rats deciding between differently delayed or sized rewards. Nature neuroscience, 10(12):1615-1624, 2007.
[27] William R Thompson. On the likelihood that one unknown probability exceeds another in view of the evidence of two samples. Biometrika, 25(3/4):285-294, 1933.
[28] Daniel Russo, Benjamin Van Roy, Abbas Kazerouni, Ian Osband, and Zheng Wen. A tutorial on Thompson sampling. arXiv preprint arXiv:1707.02038, 2017.
[29] Edward Vul, Noah Goodman, Thomas L Griffiths, and Joshua B Tenenbaum. One and done? Optimal decisions from very few samples. Cognitive science, 38(4):599-637, 2014.
[30] Richard P Heitz. The speed-accuracy tradeoff: history, physiology, methodology, and behavior. Frontiers in neuroscience, 8:150, 2014.
[31] Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8(3-4):229-256, 1992.
[32] John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. High-dimensional continuous control using generalized advantage estimation. arXiv preprint arXiv:1506.02438, 2015.
[33] Diederik P Kingma and Max Welling. Auto-encoding variational Bayes. arXiv preprint arXiv:1312.6114, 2013.
[34] Nikolaus Hansen. The CMA evolution strategy: A tutorial. arXiv preprint arXiv:1604.00772, 2016.

---

#### Page 12

[35] Nikolaus Hansen, Youhei Akimoto, and Petr Baudis. CMA-ES/pycma on Github. Zenodo, DOI:10.5281/zenodo.2559634, February 2019. URL https://doi.org/10.5281/zenodo.2559634.
[36] Harm Van Seijen, Hado Van Hasselt, Shimon Whiteson, and Marco Wiering. A theoretical and empirical analysis of Expected Sarsa. In 2009 ieee symposium on adaptive dynamic programming and reinforcement learning, pages 177-184. IEEE, 2009.
[37] Leslie G Ungerleider, Julien Doyon, and Avi Karni. Imaging brain plasticity during motor skill learning. Neurobiology of learning and memory, 78(3):553-564, 2002.
[38] Eran Dayan and Leonardo G Cohen. Neuroplasticity subserving motor skill learning. Neuron, 72(3): $443-454,2011$.
[39] Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba. OpenAI gym, 2016.
[40] Quentin JM Huys, Niall Lally, Paul Faulkner, Neir Eshel, Erich Seifritz, Samuel J Gershman, Peter Dayan, and Jonathan P Roiser. Interplay of approximate planning strategies. Proceedings of the National Academy of Sciences, 112(10):3098-3103, 2015.
[41] Jan Drugowitsch, Gregory C DeAngelis, Dora E Angelaki, and Alexandre Pouget. Tuning the speedaccuracy trade-off to maximize reward rate in multisensory decision-making. Elife, 4:e06678, 2015.
[42] Danielle Panoz-Brown, Vishakh Iyer, Lawrence M Carey, Christina M Sluka, Gabriela Rajic, Jesse Kestenman, Meredith Gentry, Sydney Brotheridge, Isaac Somekh, Hannah E Corbin, et al. Replay of episodic memories in the rat. Current Biology, 28(10):1628-1634, 2018.
[43] Anoopum S Gupta, Matthijs AA van der Meer, David S Touretzky, and A David Redish. Hippocampal replay is not a simple function of experience. Neuron, 65(5):695-705, 2010.
[44] Will Dabney, Zeb Kurth-Nelson, Naoshige Uchida, Clara Kwon Starkweather, Demis Hassabis, Rémi Munos, and Matthew Botvinick. A distributional code for value in dopamine-based reinforcement learning. Nature, pages 1-5, 2020.
[45] Farzaneh Najafi, Gamaleldin F Elsayed, Robin Cao, Eftychios Pnevmatikakis, Peter E Latham, John P Cunningham, and Anne K Churchland. Excitatory and inhibitory subnetworks are equally selective during decision-making and emerge simultaneously during learning. Neuron, 105(1):165-179, 2020.
[46] Wei Ji Ma, Jeffrey M Beck, Peter E Latham, and Alexandre Pouget. Bayesian inference with probabilistic population codes. Nature neuroscience, 9(11):1432-1438, 2006.
[47] Han Hou, Qihao Zheng, Yuchen Zhao, Alexandre Pouget, and Yong Gu. Neural correlates of optimal multisensory decision making under time-varying reliabilities with an invariant linear probabilistic population code. Neuron, 104(5):1010-1021, 2019.
[48] Dhushan Thevarajah, Ryan Webb, Christopher Ferrall, and Michael C Dorris. Modeling the value of strategic actions in the superior colliculus. Frontiers in behavioral neuroscience, 3:57, 2010.
[49] Takuro Ikeda and Okihide Hikosaka. Positive and negative modulation of motor response in primate superior colliculus by reward expectation. Journal of Neurophysiology, 98(6):3163-3170, 2007.
[50] Rubén Moreno-Bote, Jeffrey Beck, Ingmar Kanitscheider, Xaq Pitkow, Peter Latham, and Alexandre Pouget. Information-limiting correlations. Nature neuroscience, 17(10):1410, 2014.
[51] Karen Hao. Training a single ai model can emit as much carbon as five cars in their lifetimes. MIT Technology Review, 2019.
[52] Emma Strubell, Ananya Ganesh, and Andrew McCallum. Energy and policy considerations for deep learning in NLP. arXiv preprint arXiv:1906.02243, 2019.
[53] Hiroaki Kitano, Satoshi Tadokoro, Itsuki Noda, Hitoshi Matsubara, Tomoichi Takahashi, Atsuhi Shinjou, and Susumu Shimada. Robocup rescue: Search and rescue in large-scale disasters as a domain for autonomous agents research. In IEEE SMC'99 Conference Proceedings. 1999 IEEE International Conference on Systems, Man, and Cybernetics (Cat. No. 99CH37028), volume 6, pages 739-743. IEEE, 1999.
[54] Javaid Khurshid and Hong Bing-Rong. Military robots-a glimpse from today and tomorrow. In ICARCV 2004 8th Control, Automation, Robotics and Vision Conference, 2004., volume 1, pages 771-777. IEEE, 2004.
[55] Patrick Lin, George Bekey, and Keith Abney. Autonomous military robotics: Risk, ethics, and design. Technical report, California Polytechnic State Univ San Luis Obispo, 2008.