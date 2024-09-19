# Image-action-recognition# Image-action-recognition using Deep Learning

## Project Overview
This project develops a deep convolutional neural network (CNN) to identify human actions from still images. The model predicts both the action performed (from 40 categories) and whether there is more than one person in the image.

## Repository Contents
- `notebooks/`: Jupyter notebooks containing the main analysis and model development
- `models/`: Trained model files
- `predictions/`: CSV file with predictions for the future dataset

## Files Not Included (Due to License Agreement)
- Original dataset images
- Preprocessed images
- train_data_2024.csv
- future_data_2024.csv
- Any metadata files derived from the original dataset

## Setup and Installation
1. Clone this repository
2. Install required dependencies:
3.Due to licensing restrictions, the original dataset is not included. To run this project, you need to obtain the dataset from the course instructors 

## Running the Project
1. Open the main notebook in the `notebooks/` directory
2. Ensure all data files are in place as described in the Setup section
3. Run the cells in order to preprocess data, train the model, and generate predictions.

## Model Architecture
Model Selection: My approach to human action recognition (HAR) utilizes transfer learning with two pre-trained models: VGG16 and ResNet50. This choice is inspired by the research of SaiRamesh et al. (2024) , who demonstrated the effectiveness of ensemble learning in HAR tasks, achieving an impressive 98% accuracy using four models.

VGG16: Developed by Simonyan and Zisserman (2014), VGG16 is known for its simplicity and depth. Its uniform architecture of 3x3 convolutional layers stacked on top of each other makes it an excellent feature extractor.

ResNet50: Introduced by He et al. (2015), ResNet50 addresses the vanishing gradient problem in deep networks through residual connections. This allows for training of very deep networks, which can capture more complex features.

While the referenced study used four models, I have adapted my approach to use two models due to computational constraints. This modification still allows me to leverage the benefits of ensemble learning while balancing resource utilization.

Reference: SaiRamesh, L., Dhanalakshmi, B., & Selvakumar, K. (2024). Human Activity Recognition Through Images Using a Deep Learning Approach. St. Joseph's Institute of Technology, B.S. Abdur Rahman Crescent Institute of Science and Technology, National Institute of Technology Trichy. https://doi.org/10.21203/rs.3.rs-4443695/v1

Custom Layers and Their Justification After the base models, I have added custom layers to adapt the pre-trained networks to the specific HAR task:

Global Average Pooling (GAP):
Reduces spatial dimensions, decreasing the number of parameters
Helps in handling different input image sizes and mitigates overfitting by reducing the total number of parameters
Dense Layer (256 units):

Adds non-linearity and learns high-level features specific to our HAR task
ReLU activation introduces non-linearity without the vanishing gradient problem
Dropout Layer:

Prevents overfitting by randomly setting 50% of input units to 0 during training
Improves generalization by forcing the network to learn with different neurons
Output Layers:



Two separate output layers for our multi-task learning approach:
class_output: Softmax activation for multi-class action classification
morethanoneperson_output: Sigmoid activation for binary classification of multiple person presence
Optimization and Learning I use the Adam optimizer with different learning

I use the Adam optimizer with different learning pythonCopylearning_rates = [1e-4, 1e-5]

This approach allows for fine-tuning of each model independently, acknowledging that different architectures may require different learning dynamics.


Loss Functions and Their Justification

Sparse Categorical Crossentropy for 'class_output':Used for multi-class classification problems where classes are mutually exclusive 'Sparse' version is used because labels are integers (not one-hot encoded)
Efficient for problems with a large number of classes (in our case, 40 action classes)
Binary Crossentropy for 'morethanoneperson_output':Used for binary classification problems

Measures the performance of a classification model whose output is a probability value between 0 and 1

Goal:By adapting the ensemble learning approach from SaiRamesh et al. (2024) and incorporating transfer learning with VGG16 and ResNet50, I aim to achieve accuracy Of atleast 75 percent in HAR task while working within my computational constraints.

## Enhancements to Human Action Recognition Model:
Addressing Performance Issues Based on the initial results where VGG16 underperformed and the ensemble approach didn't meet expectations, I 've implemented several strategic changes to improve model performance. Here's an analysis of the key modifications:

Replaced VGG16 with EfficientNetB0
Justification: EfficientNetB0 is known for its balance of efficiency and accuracy, potentially offering better performance than VGG16 for this task.
This approach was inspired by the methods outlined in the research paper, which utilized EfficientNetB0 as part of an ensemble learning strategy to improve classification performance
Reference:Hojat Asgarian Dehkordi, Ali Soltani Nezhad, Seyed Sajad Ashrafi, Shahriar B. Shokouhi. "Still Image Action Recognition Using Ensemble Learning." 2021 7th International Conference on Web Research (ICWR), 2021, pp. 125-129. IEEE. DOI: 10.1109/ICWR51868.2021.9443021.

Regularization Techniques:

Added L2 regularization to Dense layers:
Justification: L2 regularization helps prevent overfitting by penalizing large weights, encouraging the model to learn more generalizable features.
Data Augmentation

Justification: Data augmentation increases the diversity of the training data, helping the model generalize better and reducing overfitting.
Random Horizontal Flip: Justification: Many human actions look similar when mirrored horizontally (e.g., walking left vs. right). By randomly flipping images horizontally, the model learns to recognize actions regardless of the direction they are performed in.
Random Rotation: Justification: Human actions can be performed at various angles, and the camera capturing the action might not always be perfectly aligned
Random Zoom: Justification: The scale at which actions are observed can vary depending on the distance between the camera and the subject. By randomly zooming in and out, the model learns to recognize actions from different distances, enhancing its ability to generalize across images where the subject might appear larger or smaller.
Enhanced Evaluation Metrics:

Added precision, recall, and AUC metrics for both tasks
Precision: Justification: Precision measures the accuracy of the positive predictions made by the model. For human action recognition, this metric is important because it ensures that when the model predicts a specific action class, it is correct a high proportion of the time.
Recall: Justification: Recall measures the model's ability to correctly identify all relevant instances of a class. In the context of human action recognition, high recall ensures that the model does not miss actions that are present in the images, which is crucial for applications where it is more important to capture all instances of an action

AUC (Area Under the Curve): Justification: AUC provides an aggregate measure of performance across all classification thresholds, offering a balanced view of the model's ability to distinguish between classes. It is especially useful when dealing with imbalanced datasets, which is common in action recognition tasks where some actions might be more frequent than others

Loss Weighting:

Implemented loss weighting in model compilation: Justification: This allows for balancing the importance of the two tasks during training, potentially improving overall performance.

Image Loading and Preprocessing: Improved image loading function to handle tensor inputs and ensure consistent shape: Justification: This ensures robust handling of input data and consistent image dimensions.

These enhancements address several potential issues in the original implementation:

The model architecture change tackles the poor performance of VGG16. Regularization and data augmentation target overfitting issues. Enhanced metrics provide a more nuanced understanding of model performance. Loss weighting allows for better balancing of the multi-task learning problem. Improved data handling ensures consistency in input processing.

The model architecture change tackles the poor performance of VGG16.

Regularization and data augmentation target overfitting issues.

Enhanced metrics provide a more nuanced understanding of model performance and Loss weighting allows for better balancing of the multi-task learning problem.

## Performance
Significant Improvement from VGGNet16 to EfficientNetB0 The replacement of VGGNet16 with EfficientNetB0 (Model 1) has led to a substantial improvement in performance:

Class Accuracy: Model 1 achieved 77% accuracy, a significant increase from the previous VGGNet16 model's poor performance of around 5%.
MoreThanOnePerson Accuracy: Improved to 78.49%, up from about 63% in the previous iteration. Overall Performance: Model 1 (EfficientNetB0) outperformed Model 2 (ResNet50) in most metrics, demonstrating the effectiveness of this architecture change.
Impact of Enhancements on Model Performance The implemented enhancements have collectively contributed to improved model performance:

Regularization: The gap between training and validation accuracy has decreased, especially for Model 1, suggesting that L2 regularization has effectively combated overfitting.

Loss Weighting: The balanced performance between class prediction and MoreThanOnePerson detection indicates that loss weighting has helped in managing the multi-task learning problem effectively.

Data Augmentation: The learning curves show smoother progression and better generalization, indicating that data augmentation has helped in reducing overfitting.

Success of the Ensemble Approach The ensemble model has successfully leveraged the strengths of both individual models:

Class Accuracy: The ensemble achieved 77.33% accuracy, surpassing both individual models (77% for Model 1 and 73.56% for Model 2).
MoreThanOnePerson Accuracy: A notable improvement to 82.78%, significantly higher than both individual models.
AUC Scores: The ensemble model achieved the highest AUC scores (0.9868 for Class and 0.8695 for MoreThanOnePerson), indicating superior overall performance.
Achievement of Project Goals The primary goal of achieving at least 75% accuracy has been successfully met and exceeded

Additional Observations
Precision and Recall: The ensemble model shows balanced precision and recall scores, indicating robust performance across different classes and scenarios.

AUC Scores: High AUC scores (0.9868 for Class and 0.8695 for MoreThanOnePerson) suggest excellent discriminative ability of the model.

## Ethical Considerations
Ethical Challenges

Privacy Concerns: The technology could be used for surveillance without consent, potentially infringing on personal privacy. In this project, the ability to detect whether more than one person is present could be misused for monitoring social interactions.

Consent and Data Collection: The dataset used may raise questions about whether individuals were aware their actions were being recorded and used for AI training. Consider: Were the subjects in the training images aware of how their data would be used?

Misuse and Malicious Application: Action recognition could be used for profiling or discriminatory practices. The model's ability to classify actions could be misused to monitor and control behavior in oppressive contexts.

Accuracy and Consequences: False positives or negatives in action recognition could lead to serious consequences, especially if used in security or legal contexts. My model's 77.33% accuracy, while good, still leaves room for error that could impact individuals if used in critical applications.

Potential Biases in the Dataset

Demographic Bias: The dataset may not represent a diverse range of ethnicities, ages, or body types, leading to lower accuracy for underrepresented groups. Question to consider: Does this dataset include a balanced representation of different demographics?

Environmental Bias: If the dataset primarily contains images from certain types of environments (e.g., indoor settings), the model may perform poorly in other contexts.

Action Class Imbalance: The dataset might have an uneven distribution of action classes, potentially leading to biased performance favoring over-represented actions.

Contextual Bias: The model may struggle with actions that are ambiguous without broader context, potentially leading to misclassifications.

Mitigation Strategies

Diverse Dataset Curation: Ensure the training data represents a wide range of demographics, cultures, and environments.

Transparency : Clearly communicate the limitations and potential biases of the model to end-users.

Continuous Monitoring: Regularly assess the model's performance across different groups and contexts to identify and address emerging biases.

Ethical Guidelines: Develop and adhere to strict ethical guidelines for the development and deployment of action recognition technology.

Privacy-Preserving Techniques: Implement techniques like federated learning or differential privacy to enhance data protection.

Stakeholder Engagement: Involve ethicists, legal experts, and diverse community representatives in the development and deployment process.

## Author
Raj Sanshi

## Acknowledgements
This project was completed as part of the COSC 2779/2972 Deep Learning course at RMIT University. The original dataset is from the Stanford 40 Actions Dataset.

## License
The code in this repository is for academic purposes only and is not licensed for redistribution or commercial use. The dataset used in this project is subject to its own licensing terms and is not included in this repository.