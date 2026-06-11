# Amortized Bayesian Workflow - Backmatter

---

## Acknowledgments

We thank Alexander Fengler for the help with PyMC implementation of the drift- diffusion model. CL and LA were supported by the Research Council of Finland (grants number 356498 and 358980 to LA). AV acknowledges the Research Council of Finland Flagship program: Finnish Center for Artificial Intelligence, and Academy of Finland project 340721. STR is supported by the National Science Foundation under Grant No. 2448380. MS and PB acknowledge support of Cyber Valley Project CyVy- RF- 2021- 16, the DFG under Germany's Excellence Strategy – EXC- 2075 – 390740016 (the Stuttgart Cluster of Excellence SimTech). PB additionally acknowledges the support of DFG Collaborative Research Cluster 391 “Spatio- Temporal Statistics for the Transition of Energy and Transport” – 520388526. MS acknowledges travel support from the European Union’s Horizon 2020 research and innovation programme under grant agreements No 951847 (ELISE) and No 101070617 (ELSA), and support from the Aalto Science- IT project. The authors wish to thank the Finnish Computing Competence Infrastructure (FCCI) for supporting this project with computational and data storage resources. The authors also acknowledge the research environment provided by ELLIS Institute Finland.

## References

Luigi Acerbi. Variational Bayesian Monte Carlo with noisy likelihoods. Advances in Neural Information Processing Systems, 33:8211–8222, 2020.

Michael Arbel, Alex Matthews, and Arnaud Doucet. Annealed flow transport Monte Carlo. In Proceedings of the 38th International Conference on Machine Learning, pp. 318–330. PMLR, July 2021.

Sebastian Bieringer, Anja Butter, Theo Heimel, Stefan Höche, Ullrich Köthe, Tilman Plehn, and Stefan T Radev. Measuring qcd splittings with invertible networks. SciPost Physics, 10(6):126, 2021.

Eli Bingham, Jonathan P. Chen, Martin Jankowiak, Fritz Obermeyer, Neeraj Pradhan, Theofanis Karaletsos, Rohit Singh, Paul Szerlip, Paul Horsfall, and Noah D. Goodman. Pyro: Deep universal probabilistic programming. Journal of Machine Learning Research, 20(28):1–6, 2019. ISSN 1533- 7928.

Jan Boelts, Michael Deistler, Manuel Gloeckler, Álvaro Tejero- Cantero, Jan- Matthias Lueckmann, Guy Moss, Peter Steinbach, Thomas Moreau, Fabio Muratore, Julia Linhart, Conor Durkan, Julius Vetter, Benjamin Kurt Miller, Maternus Herold, Abolfazl Ziaeemehr, Matthijs Pals, Theo Gruner, Sebastian Bischoff, Nastya Krouglova, Richard Gao, Janne K. Lappalainen, Bálint Mucsányi, Felix Pei, Auguste Schulz, Zinovia Stefanidi, Pedro Rodrigues, Cornelius Schröder, Farid Abu Zaid, Jonas Beck, Jaivardhan Kapoor, David S. Greenberg, Pedro J. Gonçalves, and Jakob H. Macke. Sbi reloaded: A toolkit for simulation- based inference workflows. Journal of Open Source Software, 10(108):7754, April 2025. ISSN 2475- 9066. doi: 10.21105/joss.07754.

James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Chris Leary, Dougal Maclaurin, George Necula, Adam Paszke, Jake VanderPlas, Skye Wanderman- Milne, and Qiao Zhang. JAX: composable transformations of Python+NumPy programs, 2018. URL http://github.com/google/jax.

Alberto Cabezas and Christopher Nemeth. Transport elliptical slice sampling. In Proceedings of The 26th International Conference on Artificial Intelligence and Statistics, pp. 3664- 3676. PMLR, April 2023.

Alberto Cabezas, Louis Sharrock, and Christopher Nemeth. Markovian flow matching: Accelerating MCMC with continuous normalizing flows. Advances in Neural Information Processing Systems, 37:104383- 104411, December 2024.

Colin Caprani. Generalized extreme value distribution. https://www.pymc.io/projects/examples/en/latest/case_studies/GEV.html, 2021. PyMC Examples. Accessed: 2025- 05- 13.

Bob Carpenter, Andrew Gelman, Matthew D Hoffman, Daniel Lee, Ben Goodrich, Michael Betancourt, Marcus Brubaker, Jiqiang Guo, Peter Li, and Allen Riddell. Stan: A probabilistic programming language. Journal of statistical software, 76(1), 2017.

Paul E Chang, Nasrulloh Loka, Daolang Huang, Ulpu Remes, Samuel Kaski, and Luigi Acerbi. Amortized probabilistic conditioning for optimization, simulation and inference. In Proceedings of the International Conference on Artificial Intelligence and Statistics (AISTATS). PMLR, 2025.

Gang Chen, Paul- Christian Burkner, Paul A. Taylor, Zhihao Li, Lijun Yin, Daniel R. Glen, Joshua Kinnison, Robert W. Cox, and Luiz Pessoa. An integrative Bayesian approach to matrix- based analysis in neuroimaging. Human Brain Mapping, 40(14):4072- 4090, 2019. doi: 10.1002/hbm.24686.

François Chollet et al. Keras. https://keras.io, 2015.

Kyle Cranmer, Johann Brehmer, and Gilles Louppe. The frontier of simulation- based inference. Proceedings of the National Academy of Sciences, 2020.

Maximilian Dax, Stephen R. Green, Jonathan Gair, Michael Pürrer, Jonas Wildberger, Jakob H. Macke, Alessandra Buonanno, and Bernhard Schölkopf. Neural importance sampling for rapid and reliable gravitational- wave inference. Phys. Rev. Lett., 130:171403, Apr 2023. doi: 10.1103/PhysRevLett.130.171403. URL https://link.aps.org/doi/10.1103/PhysRevLett.130.171403.

Arnaud Delaunoy, Joeri Hermans, François Rozet, Antoine Wehenkel, and Gilles Louppe. Towards reliable simulation- based inference with balanced neural ratio estimation. Advances in Neural Information Processing Systems, 35:20025- 20037, December 2022.

Akash Kumar Dhaka, Alejandro Catalina, Manushi Welandawe, Michael R Andersen, Jonathan Huggins, and Aki Vehtari. Challenges and opportunities in high dimensional variational inference. In Advances in Neural Information Processing Systems, volume 34, pp. 7787- 7798. Curran Associates, Inc., 2021.

Joshua V. Dillon, Ian Langmore, Dustin Tran, Eugene Brevdo, Srinivas Vasudevan, Dave Moore, Brian Patton, Alex Alemi, Matt Hoffman, and Rif A. Saurous. TensorFlow distributions, 2017.

Lars Dingeldein, David Silva- Sánchez, Luke Evans, Edoardo D'Imprima, Nikolaus Grigorieff, Roberto Covino, and Pilar Cossio. Amortized template- matching of molecular conformations from cryo- electron microscopy images using simulation- based inference. bioRxiv, pp. 2024.07.23.604154, 2024. doi: 10.1101/2024.07.23.604154. URL http://biorxiv.org/content/early/2024/07/31/2024.07.23.604154.abstract.

Conor Durkan, Artur Bekasov, Iain Murray, and George Papamakarios. Neural spline flows. Advances in neural information processing systems, 32, 2019.

Lasse Elsemüller, Hans Olischläger, Marvin Schmitt, Paul- Christian Bürkner, Ullrich Koethe, and Stefan T. Radev. Sensitivity- aware amortized Bayesian inference. Transactions on Machine Learning Research, 2024.

Lasse Elsemüller, Valentin Pratz, Mischa von Krause, Andreas Voss, Paul- Christian Bürkner, and Stefan T. Radev. Does unsupervised domain adaptation improve the robustness of amortized Bayesian inference? a systematic evaluation, 2025.

Alexander Fengler, Yang Xu, Bera Krishn, Aisulu Omar, and Michael J. Frank. HSSM: A generalized toolbox for hierarchical Bayesian estimation of computational models in cognitive neuroscience. Manuscript in preparation, 2025. URL https://github.com/lnccbrown/HSSM.

D. Foreman-Mackey, D. W. Hogg, D. Lang, and J. Goodman. emcee: The MCMC hammer. PASP, 125: 306-312, 2013. doi: 10.1086/670067.

David T. Frazier, Ryan Kelly, Christopher Drovandi, and David J. Warne. The statistical accuracy of neural posterior and likelihood estimation, 2024.

Marylou Gabrié, Grant M. Rotskoff, and Eric Vanden-Eijnden. Adaptive Monte Carlo augmented with normalizing flows. Proceedings of the National Academy of Sciences, 119(10):e2109420119, March 2022. doi: 10.1073/pnas.2109420119.

Tomas Geffner, George Papamakarios, and Andriy Mnih. Compositional score modeling for simulation- based inference. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (eds.), Proceedings of the 40th international conference on machine learning, volume 202 of Proceedings of machine learning research, pp. 11098- 11116. PMLR, 2023. URL https://proceedings.mlr.press/v202/geffner23a.html.

Andrew Gelman and Donald B. Rubin. Inference from Iterative Simulation Using Multiple Sequences. Statistical Science, 7(4):457- 472, November 1992. ISSN 0883- 4237, 2168- 8745. doi: 10.1214/ss/1177011136.

Andrew Gelman, John B Carlin, Hal S Stern, David B Dunson, Aki Vehtari, and Donald B Rubin. Bayesian Data Analysis. Chapman & Hall/CRC, Philadelphia, PA, 3 edition, 2013.

Andrew Gelman, Aki Vehtari, Daniel Simpson, et al. Bayesian workflow. arXiv preprint, 2020.

J.- P. George, P.- C. Bürkner, T. G. M. Sanders, M. Neumann, C. Cammalleri, J. V. Vogt, and M. Lang. Long- term forest monitoring reveals constant mortality rise in European forests. Plant Biology, 24(7): 1108- 1119, 2022. doi: 10.1111/plb.13469.

Amin Ghaderi- Kangavari, Jamal Amani Rad, and Michael D. Nunez. A general integrative neurocognitive modeling framework to jointly describe EEG and decision- making on single trials. Computational Brain & Behavior, 6(3):317- 376, 2023. ISSN 2522- 0861, 2522- 087X. doi: 10.1007/s42113- 023- 00167- 4. URL https://link.springer.com/10.1007/s42113- 023- 00167- 4.

Manuel Glöckler, Michael Deistler, and Jakob H. Macke. Variational methods for simulation- based inference. In International Conference on Learning Representations, 2022.

Manuel Glöckler, Michael Deistler, Christian Weilbach, Frank Wood, and Jakob H Macke. All- in- one simulation- based inference. In Proceedings the International Conference on Machine Learning (ICML), pp. 15735- 15766, 2024.

Tilmann Gneiting and Adrian E Raftery. Strictly proper scoring rules, prediction, and estimation. Journal of the American statistical Association, 102(477):359- 378, 2007.

Pedro J Gonçalves, Jan- Matthias Lueckmann, Michael Deistler, Marcel Nonnenmacher, Kaan Öcal, Giacomo Bassetto, Chaitanya Chintaluri, William F Podlaski, Sara A Haddad, Tim P Vogels, David S Greenberg, and Jakob H Macke. Training deep neural density estimators to identify mechanistic models of neural dynamics. eLife, 9:e56261, September 2020. ISSN 2050- 084X. doi: 10.7554/eLife.56261.

David Greenberg, Marcel Nonnenmacher, and Jakob Macke. Automatic posterior transformation for likelihood- free inference. In International Conference on Machine Learning, 2019.

Louis Grenioux, Alain Oliviero Durmus, Eric Moulines, and Marylou Gabrié. On sampling with approximate transport maps. In Proceedings of the 40th International Conference on Machine Learning, pp. 11698- 11733. PMLR, July 2023.

Joeri Hermans, Arnaud Delaunoy, François Rozet, Antoine Wehenkel, Volodimir Begy, and Gilles Louppe. A crisis in simulation- based inference? Beware, your posterior approximations can be unfaithful. Transactions on Machine Learning Research, September 2022. ISSN 2835- 8856.

Matthew Hoffman, Pavel Sountsov, Joshua V. Dillon, Ian Langmore, Dustin Tran, and Srinivas Vasudevan. NeuTra- lizing bad geometry in Hamiltonian Monte Carlo using neural transport, March 2019.

Matthew Hoffman, Alexey Radul, and Pavel Sountsov. An adaptive- MCMC scheme for setting trajectory lengths in Hamiltonian Monte Carlo. In Arindam Banerjee and Kenji Fukumizu (eds.), Proceedings of The 24th International Conference on Artificial Intelligence and Statistics, volume 130 of Proceedings of Machine Learning Research, pp. 3907- 3915. PMLR, 13- 15 Apr 2021.

Matthew D. Hoffman and Andrew Gelman. The No- U- Turn Sampler: Adaptively setting path lengths in Hamiltonian Monte Carlo. Journal of Machine Learning Research, 15(47):1593- 1623, 2014.

Daolang Huang, Ayush Bharti, Amauri Souza, Luigi Acerbi, and Samuel Kaski. Learning robust statistics for simulation- based inference under model misspecification, 2023.

Ravin Kumar, Colin Carroll, Ari Hartikainen, and Osvaldo Martin. Arviz a unified library for exploratory analysis of bayesian models in python. Journal of Open Source Software, 4(33):1143, 2019. doi: 10.21105/ joss.01143. URL https://doi.org/10.21105/joss.01143.

Nils C. Landmeyer, Paul- Christian Burkner, Heinz Wiendl, Tobias Ruck, Hans- Peter Hartung, Heinz Holling, and Meuth. Disease- modifying treatments and cognition in relapsing- remitting multiple sclerosis: A meta- analysis. Neurology, 94(22):2373- 2383, 2020. doi: 10.1212/WNL.0000000000009522.

Alexander Lavin, Hector Zenil, Brooks Paige, et al. Simulation intelligence: Towards a new generation of scientific methods. arXiv preprint, 2021.

Juho Lee, Yoonho Lee, Jungtaek Kim, Adam Kosiorek, Seungjin Choi, and Yee Whye Teh. Set transformer: A framework for attention- based permutation- invariant neural networks. In Proceedings of the 36th International Conference on Machine Learning, pp. 3744- 3753, 2019.

Chengkun Li, Bobby Huggins, Petrus Mikkola, and Luigi Acerbi. Normalizing flow regression for Bayesian inference with offline likelihood evaluations. In 7th Symposium on Advances in Approximate Bayesian Inference, 2025.

Julia Linhart, Alexandre Gramfort, and Pedro L. C. Rodrigues. L- C2ST: Local diagnostics for posterior approximations in simulation- based inference. In Thirty- seventh Conference on Neural Information Processing Systems, 2023.

Julia Linhart, Gabriel Victorino Cardoso, Alexandre Gramfort, Sylvain Le Corff, and Pedro L. C. Rodrigues. Diffusion posterior sampling for simulation- based inference in tall data settings, June 2024. URL http://arxiv.org/abs/2404.07593. arXiv:2404.07593 [cs, stat].

Yaron Lipman, Ricky T. Q. Chen, Heli Ben- Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations, 2023.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019.

Jan- Matthias Lueckmann, Jan Boelts, David Greenberg, Pedro Goncalves, and Jakob Macke. Benchmarking simulation- based inference. In Proceedings of The 24th International Conference on Artificial Intelligence and Statistics, pp. 343- 351. PMLR, March 2021.

Tuulia Malén, Tomi Karjalainen, Janne Isojärvi, Aki Vehtari, Paul- Christian Bürkner, Vesa Putkinen, Valtteri Kaasinen, Jarmo Hietala, Pirjo Nuutila, Juha Rinne, and Lauri Nummenmaa. Atlas of type 2 dopamine receptors in the human brain: Age and sex dependent variability in a large PET cohort. NeuroImage, 255: 119149, 2022. doi: 10.1016/j.neuroimage.2022.119149.

Charles C. Margossian, Matthew D. Hoffman, Pavel Sountsov, Lionel Riou- Durand, Aki Vehtari, and Andrew Gelman. Nested \(\hat{R}\) : Assessing the convergence of Markov chain Monte Carlo when running many short chains. Bayesian Analysis, pp. 1 - 28, 2024. doi: 10.1214/24- BA1453. URL https://doi.org/10.1214/24- BA1453.

Laurence Illing Midgley, Vincent Stimper, Gregor N. C. Simm, Bernhard Schölkopf, and José Miguel Hernández- Lobato. Flow annealed importance sampling bootstrap. In The Eleventh International Conference on Learning Representations, September 2022.

Aayush Mishra, Daniel Habermann, Marvin Schmitt, Stefan T. Radev, and Paul- Christian Bürkner. Robust amortized Bayesian inference with self- consistency losses on unlabeled data, 2025.

Sarthak Mittal, Niels Leif Bracher, Guillaume Lajoie, Priyank Jaini, and Marcus Brubaker. Amortized In- Context Bayesian Posterior Estimation, February 2025.

Martin Modrák, Angie H. Moon, Shinyoung Kim, Paul Bürkner, Niko Huurre, Kateřina Faltejsková, Andrew Gelman, and Aki Vehtari. Simulation- based calibration checking for Bayesian computation: The choice of test quantities shapes sensitivity. Bayesian Analysis, 20(2):461- 488, June 2025. ISSN 1936- 0975, 1931- 6690. doi: 10.1214/23- BA1404.

Samuel Müller, Noah Hollmann, Sebastian Pineda Arango, Josif Grabocka, and Frank Hutter. Transformers can do Bayesian inference. In International Conference on Learning Representations (ICLR), 2022.

Radford M. Neal. Slice sampling. The Annals of Statistics, 31(3):705- 767, June 2003. ISSN 0090- 5364, 2168- 8966. doi: 10.1214/aos/1056562461.

Radford M. Neal. MCMC using Hamiltonian dynamics. In Handbook of Markov Chain Monte Carlo. Chapman and Hall/CRC, 2011. ISBN 978- 0- 429- 13850- 8.

Abril- Pla Oriol, Andreani Virgile, Carroll Colin, Dong Larry, Fonnesbeck Christopher J., Kochurov Maxim, Kumar Ravin, Lao Jupeng, Luhmann Christian C., Martin Osvaldo A., Osthege Michael, Vieira Ricardo, Wiecki Thomas, and Zinkov Robert. PyMC: A modern and comprehensive probabilistic programming framework in Python. PeerJ Computer Science, 9:e1516, 2023. doi: 10.7717/peerj- cs.1516.

Rafael Orozco, Ali Siahkoohi, Mathias Louboutin, and Felix J Herrmann. Aspire: iterative amortized posterior inference for Bayesian inverse problems. Inverse Problems, 41(4), 2025.

Lorenzo Pacchiardi and Ritabrata Dutta. Likelihood- free inference with generative neural networks via scoring rule minimization, May 2022. arXiv:2205.15784.

George Papamakarios, Eric Nalisnick, Danilo Jimenez Rezende, Shakir Mohamed, and Balaji Lakshminarayanan. Normalizing flows for probabilistic modeling and inference. Journal of Machine Learning Research, 22(57):1- 64, 2021.

Matthew D. Parno and Youssef M. Marzouk. Transport map accelerated Markov chain Monte Carlo. SIAM/ASA Journal on Uncertainty Quantification, 6(2):645- 682, January 2018. doi: 10.1137/17M1134640.

Du Phan, Neeraj Pradhan, and Martin Jankowiak. Composable effects for flexible and accelerated probabilistic programming in numpyro. arXiv preprint arXiv:1912.11554, 2019.

Stefan T Radev, Ulf K Mertens, Andreas Voss, Lynton Ardizzone, and Ullrich Kothe. Bayesflow: Learning complex stochastic models with invertible neural networks. IEEE transactions on neural networks and learning systems, 2020.

Stefan T Radev, Frederik Graw, Simiao Chen, Nico T Mutters, Vanessa M Eichel, Till Barnighausen, and Ullrich Kothe. Outbreakflow: Model- based Bayesian inference of disease outbreak dynamics with invertible neural networks and its application to the covid- 19 pandemics in germany. PLoS computational biology, 2021.

Stefan T. Radev, Marvin Schmitt, Lukas Schumacher, Lasse Elsemüller, Valentin Pratz, Yannik Schälte, Ullrich Kothe, and Paul- Christian Bürkner. BayesFlow: Amortized Bayesian workflows with neural networks. Journal of Open Source Software, 8(89):5702, 2023. doi: 10.21105/joss.05702. URL https://doi.org/10.21105/joss.05702.

Roger Ratcliff and Gail McKoon. The diffusion decision model: Theory and data for two- choice decision tasks. Neural Computation, 20(4):873- 922, 2008.

Aura Raulo, Paul- Christian Bürkner, Jarrah Dale, English, Curt Lamberth, Josh A. Firth, and Coulson. Social and environmental transmission spread different sets of gut microbes in wild mice. bioRxiv preprint, 2023. doi: 10.1101/2023.07.20.549849.

Danilo Rezende and Shakir Mohamed. Variational inference with normalizing flows. In Francis Bach and David Blei (eds.), Proceedings of the 32nd International Conference on Machine Learning, volume 37 of Proceedings of Machine Learning Research, pp. 1530- 1538, Lille, France, 07- 09 Jul 2015. PMLR. URL https://proceedings.mlr.press/v37/rezende15. html.

Donald B. Rubin. Using the SIR algorithm to simulate posterior distributions. In Bayesian statistics 3. Proceedings of the third Valencia international meeting, 1- 5 June 1987, pp. 395- 402. Clarendon Press, 1988.

Teemu Säilynoja, Paul- Christian Bürkner, and Aki Vehtari. Graphical test for discrete uniformity and its applications in goodness- of- fit evaluation and multiple sample comparison. Statistics and Computing, 32(2):1- 21, 2022.

Tim Salimans, Diederik Kingma, and Max Welling. Markov chain Monte Carlo and variational inference: Bridging the gap. In Proceedings of the 32nd International Conference on Machine Learning, pp. 1218- 1226. PMLR, June 2015.

Marvin Schmitt, Paul- Christian Bürkner, and Kothe. Detecting model misspecification in amortized Bayesian inference with neural networks. Proceedings of the German Conference on Pattern Recognition (GCPR), 2023.

Marvin Schmitt, Desi R. Ivanova, Daniel Habermann, Ullrich Kothe, Paul- Christian Bürkner, and Stefan T. Radev. Leveraging self- consistency for data- efficient amortized Bayesian inference. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp (eds.), Proceedings of the 41st international conference on machine learning, volume 235 of Proceedings of machine learning research, pp. 43723- 43741. PMLR, 2024a. URL https://proceedings.mlr.press/v235/schmitt24a.html.

Marvin Schmitt, Valentin Pratz, Ullrich Kothe, Paul- Christian Bürkner, and Stefan T. Radev. Consistency models for scalable and fast simulation- based inference. In Proceedings of the 38th international conference on neural information processing systems, 2024b. URL http://arxiv.org/abs/2312.05440.

Ilona Schneider, Harald Kugel, Ronny Redlich, Dominik Grotegeld, Christian Bürger, Paul- Christian Bürkner, Nils Opel, Katharina Dohm, Dario Zaremba, Susanne Meinert, et al. Association of serotonin transporter gene AluJb methylation with major depression, amygdala responsiveness, 5- HTTLPR/rs25531 polymorphism, and stress. Neuropsychopharmacology, 43(6):1308- 1316, 2018. doi: 10.1038/npp.2017.273.

Heiko H. Schütt, Stefan Harmeling, Jakob H. Macke, and Felix A. Wichmann. Painfree and accurate Bayesian estimation of psychometric functions for (potentially) overdispersed data. Vision Research, 122:105- 123, May 2016. ISSN 0042- 6989. doi: 10.1016/j.visres.2016.02.002.

Fiona M. Seaton, David A. Robinson, Don Monteith, Inma Lebron, Paul- Christian Burkner, Sam Tomlinson, Bridget A. Emmett, and Simon M. Smart. Fifty years of reduction in sulphur deposition drives recovery in soil pH and plant communities. Journal of Ecology, 111(2):464- 478, 2023. doi: 10.1111/1365- 2745.14039.

Louis Sharrock, Jack Simons, Song Liu, and Mark Beaumont. Sequential neural score estimation: Likelihood- free inference with conditional score based diffusion models. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp (eds.), Proceedings of the 41st international conference on machine learning, volume 235 of Proceedings of machine learning research, pp. 44565- 44602. PMLR, 2024. URL https://proceedings.mlr.press/v235/sharrock24a.html.

Ali Siahkoohi, Gabrio Rizzuti, Rafael Orozco, and Felix J. Herrmann. Reliable amortized variational inference with physics- based latent distribution correction. GEOPHYSICS, 88(3), 2023.

Yang Song, Jascha Sohl- Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score- based generative modeling through stochastic differential equations. In International conference on learning representations, 2021.

Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (eds.), Proceedings of the 40th international conference on machine learning, volume 202 of Proceedings of machine learning research, pp. 32211- 32252. PMLR, 2023. URL https://proceedings.mlr.press/v202/song23a.html.

Pavel Sountsov, Colin Carroll, and Matthew D. Hoffman. Running Markov chain Monte Carlo on modern hardware and software, November 2024.

Vladimir Starostin, Maximilian Dax, Alexander Gerlach, Alexander Hinderhofer, Álvaro Tejero- Cantero, and Frank Schreiber. Fast and reliable probabilistic reflectometry inversion with prior- amortized neural posterior estimation. Science Advances, 11(11):eadr9668, March 2025. ISSN 2375- 2548. doi: 10.1126/sciadv.adr9668.

Teemu Säilynoja, Marvin Schmitt, Paul- Christian Bürkner, and Aki Vehtari. Posterior SBC: Simulation- based calibration checking conditional on data. arXiv:2502.03279, 2025.

Sean Talts, Michael Betancourt, Daniel Simpson, Aki Vehtari, and Andrew Gelman. Validating Bayesian inference algorithms with simulation- based calibration. arXiv preprint, 2018.

The International Brain Laboratory, Valeria Aguillon- Rodriguez, Dora Angelaki, Hannah Bayer, Niccolo Bonacchi, Matteo Carandini, Fanny Cazettes, Gaelle Chapuis, Anne K Churchland, Yang Dan, Eric Dewitt, Mayo Faulkner, Hamish Forrest, Laura Haetzel, Michael Häusser, Sonja B Hofer, Fei Hu, Anup Khanal, Christopher Krasniak, Ines Laranjeira, Zachary F Mainen, Guido Meijer, Nathaniel J Miska, Thomas D Mrsic- Flogel, Masayoshi Murakami, Jean- Paul Noel, Alejandro Pan- Vazquez, Cyrille Rossant, Joshua Sanders, Karolina Socha, Rebecca Terry, Anne E Urai, Hernando Vergara, Miles Wells, Christian J Wilson, Ilana B Witten, Lauren E Wool, and Anthony M Zador. Standardized and reproducible measurement of decision- making in mice. eLife, 10:e63711, May 2021. ISSN 2050- 084X. doi: 10.7554/eLife.63711.

Panagiotis Tsilifis and Sayan Ghosh. Inverse design under uncertainty using conditional normalizing flows. In AIAA SCITECH 2022 Forum. American Institute of Aeronautics and Astronautics, January 2022. doi: 10.2514/6.2022- 0631. URL http://dx.doi.org/10.2514/6.2022- 0631.

Aki Vehtari, Andrew Gelman, Daniel Simpson, Bob Carpenter, and Paul- Christian Bürkner. Rank- normalization, folding, and localization: An improved \(\hat{R}\) for assessing convergence of MCMC (with discussion). Bayesian Analysis, 16(2), 2021. ISSN 1936- 0975. doi: 10.1214/20- ba1221. URL http://dx.doi.org/10.1214/20- BA1221.

Aki Vehtari, Daniel Simpson, Andrew Gelman, Yuling Yao, and Jonah Gabry. Pareto smoothed importance sampling. Journal of Machine Learning Research, 25(72):1- 58, 2024.

Mischa von Krause, Stefan T. Radev, and Andreas Voss. Mental speed is high until age 60 as revealed by analysis of over a million participants. Nature Human Behaviour, 6(5):700- 708, May 2022. ISSN 2397- 3374. doi: 10.1038/s41562- 021- 01282- 7.

Yu Wang, Mikolaj Kasprzak, and Jonathan H. Huggins. A targeted accuracy diagnostic for variational approximations. In Proceedings of The 26th International Conference on Artificial Intelligence and Statistics, pp. 8351- 8372. PMLR, April 2023.

Daniel Ward, Patrick Cannon, Mark Beaumont, Matteo Fasiolo, and Sebastian Schmon. Robust neural posterior estimation and statistical model criticism. Advances in Neural Information Processing Systems, 35:33845- 33859, 2022.

Antoine Wehenkel, Jens Behrmann, Andrew C. Miller, Guillermo Sapiro, Ozan Sener, Marco Cuturi Cameto, and Jörn- Henrik Jacobsen. Simulation- based inference for cardiovascular models. In NeurIPS workshop, 2024. URL https://arxiv.org/abs/2307.13918.

George Whittle, Juliusz Ziomek, Jacob Rawling, and Michael A Osborne. Distribution transformers: Fast approximate Bayesian inference with on- the- fly prior adaptation. arXiv preprint arXiv:2502.02463, 2025.

Felix A. Wichmann and N. Jeremy Hill. The psychometric function: I. Fitting, sampling, and goodness of fit. Perception & Psychophysics, 63(8):1293- 1313, November 2001. ISSN 1532- 5962. doi: 10.3758/BF03194544.

Jonas Wildberger, Maximilian Dax, Simon Buchholz, Stephen Green, Jakob H Macke, and Bernhard Schölkopf. Flow matching for scalable simulation- based inference. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine (eds.), Advances in neural information processing systems, volume 36, pp. 16837- 16864, 2023.

Kaiyuan Xu, Brian Nosek, and Anthony G. Greenwald. Psychology data from the race implicit association test on the project implicit demo website. Journal of Open Psychology Data, 2014. doi: 10.5334/jopd.ac.

Jingkang Yang, Kaiyang Zhou, Yixuan Li, and Ziwei Liu. Generalized out- of- distribution detection: A survey. International Journal of Computer Vision, 132(12):5635- 5662, 2024.

Yuling Yao and Justin Domke. Discriminative calibration: Check bayesian computation from simulations and flexible classifier. Advances in Neural Information Processing Systems, 36:36106- 36131, 2023.

Yuling Yao, Aki Vehtari, Daniel Simpson, and Andrew Gelman. Yes, but did it work?: Evaluating variational inference. In Jennifer Dy and Andreas Krause (eds.), Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pp. 5581- 5590. PMLR, 10- 15 Jul 2018.

Manzil Zaheer, Satwik Kottur, Siamak Ravanbakhsh, Barnabas Poczos, Russ R Salakhutdinov, and Alexander J Smola. Deep Sets. In Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017.

Andrew Zammit- Mangion, Matthew Sainsbury- Dale, and Raphael Huser. Neural methods for amortized inference. Annual Review of Statistics and Its Application, 12(Volume 12, 2025):311- 335, 2025. ISSN 2326- 831X. doi: 10.1146/annurev- statistics- 112723- 034123.

Lu Zhang, Bob Carpenter, Andrew Gelman, and Aki Vehtari. Pathfinder: Parallel quasi- Newton variational inference. Journal of Machine Learning Research, 23(306):1- 49, 2022. URL http://jmlr.org/papers/v23/21- 0889. html.

Lingyi Zhou, Stefan T. Radev, William H. Oliver, Aura Obreja, Zehao Jin, and Tobias Buck. Evaluating sparse galaxy simulations via out- of- distribution detection and amortized Bayesian model comparison, October 2024.

---

*Transcribed with OCR; text and equations may contain mistakes.*
