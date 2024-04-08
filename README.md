# Pose-Editing

## Pipeline
![Pipeline Image](images/pipeline.svg)

## Models Used

The following models are integral to the functionality of this project:

- **Grounded SAM**: A model that specializes in semantic segmentation and object grounding, enabling detailed analysis and manipulation of images.
- **Zero 1-to-3**: An innovative model that provides capabilities for 3D object reconstruction and manipulation from single 2D images.
- **Stable Diffusion Inpainting**: Utilized for its state-of-the-art inpainting abilities, allowing for seamless modifications and enhancements to images.

## Installation and Usage

Follow these steps to set up the environment and run the pipeline:

```bash
pip install -r requirements.txt

**Clone the GroundedSAM and Zero123 repositories into the specified directories within this project:**

git clone <GroundedSAM-Repository-URL> GroundedSAM
git clone <Zero123-Repository-URL> Zero123

**To execute the pipeline**
python run.py --image ./path/to/your/image.jpg --class "desired-object-class" --azimuth +72 --polar +0

