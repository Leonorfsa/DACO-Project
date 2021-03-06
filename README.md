# COMPUTER AIDED DIAGNOSTICS (DACO) - Final Project to be graded

Lung cancer is the deadliest cancer worldwide, but its early diagnosis can significantly in- crease the patients’ survival rate. For that purpose, the symptomatic and group risk populations (namely smokers) are screened via chest low-dose computed tomography (CT), and the resulting volumetric image is then analyzed by radiologists.
During scan assessment, radiologists search for lung nodules, which are spherical, abnormal structures that show in the lung parenchyma, either isolated or attached to the pleural wall or to lung vessels. The specialists then characterize the abnormal findings in terms of size, shape and texture. Namely, the size of the nodule and it’s texture (solid, sub-solid and non-solid) are two pivotal characteristics for referring a patient for further analysis. Computer-aided diagnosis can assist radiologists by providing a second opinion and decrease the overall diagnosis burden both in terms of workload and total procedure time. Fig. 1 shows examples of nodules of di↵erent sizes and textures.

# Goal
Given a set of cubes centered on lung nodules, the following tasks are to be accomplished: 
• develop a machine-learning approach for lung nodule segmentation;
• develop a machine-learning approach for lung nodule texture characterization;

# Evaluation
The practical project is composed of di↵erent di culty levels, and thus the maximum achievable grade depends on the cumulative complexity of the task:
1. 2D segmentation on the axial slice containing the centroid of the nodule: 14/20;
2. 2D texture prediction using the axial slice containing the centroid of the nodule: 16/20;
3. 3D segmentation: 18/20;
4. Texture prediction with 3D features: 20/20;
