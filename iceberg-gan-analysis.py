import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import time
from IPython import display
import os
import json
from datetime import datetime
import psutil
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gan_training.log'),
        logging.StreamHandler()
    ]
)

class IcebergGAN:
    def __init__(self):
        self.BUFFER_SIZE = 1604  # Size of our dataset
        self.BATCH_SIZE = 32
        self.noise_dim = 100
        self.num_examples_to_generate = 16
        
        # Metrics storage
        self.metrics = {
            'generator_loss': [],
            'discriminator_loss': [],
            'training_time': [],
            'memory_usage': [],
            'gpu_memory_usage': [],
            'quality_metrics': []
        }

    def load_and_prepare_data(self, train_data):
        logging.info("Starting data preparation...")
        start_time = time.time()
        
        images = []
        for index, row in train_data.iterrows():
            band1 = np.array(row['band_1']).reshape(75, 75)
            band2 = np.array(row['band_2']).reshape(75, 75)
            
            combined = np.dstack((band1, band2))
            combined = (combined - combined.mean()) / (combined.max() - combined.min())
            combined = combined * 2 - 1
            images.append(combined)
        
        prepared_data = np.array(images)
        
        logging.info(f"Data preparation completed. Shape: {prepared_data.shape}")
        logging.info(f"Data preparation time: {time.time() - start_time:.2f} seconds")
        
        return prepared_data

    def make_generator(self):
        model = tf.keras.Sequential([
            layers.Dense(19 * 19 * 256, use_bias=False, input_shape=(100,)),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Reshape((19, 19, 256)),

            layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(),

            layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(),

            layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(),

            layers.Conv2DTranspose(2, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh')
        ])
        return model

    def make_discriminator(self):
        model = tf.keras.Sequential([
            layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=(75, 75, 2)),
            layers.LeakyReLU(),
            layers.Dropout(0.3),

            layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
            layers.LeakyReLU(),
            layers.Dropout(0.3),

            layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'),
            layers.LeakyReLU(),
            layers.Dropout(0.3),

            layers.Flatten(),
            layers.Dense(1)
        ])
        return model

    def discriminator_loss(self, real_output, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        return cross_entropy(tf.ones_like(fake_output), fake_output)

    def get_system_metrics(self):
        metrics = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_gb': psutil.virtual_memory().used / (1024 ** 3)
        }
        return metrics

    @tf.function
    def train_step(self, images, generator, discriminator, generator_optimizer, discriminator_optimizer):
        noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(noise, training=True)

            real_output = discriminator(images, training=True)
            fake_output = discriminator(generated_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        return gen_loss, disc_loss

    def calculate_quality_metrics(self, generated_images, real_images):
        # Simple statistical comparison between real and generated images
        metrics = {
            'gen_mean': float(np.mean(generated_images)),
            'gen_std': float(np.std(generated_images)),
            'real_mean': float(np.mean(real_images)),
            'real_std': float(np.std(real_images)),
            'distribution_difference': float(np.mean(np.abs(np.mean(generated_images, axis=0) - 
                                                         np.mean(real_images, axis=0))))
        }
        return metrics

    def generate_and_save_images(self, generator, epoch, test_input):
        predictions = generator(test_input, training=False)
        
        fig = plt.figure(figsize=(10, 10))
        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow(predictions[i, :, :, 0], cmap='viridis')
            plt.axis('off')
        
        output_dir = 'generated_images'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        plt.savefig(f'{output_dir}/image_at_epoch_{epoch:04d}.png')
        plt.close()
        
        return predictions

    def train(self, dataset, epochs=50):
        generator = self.make_generator()
        discriminator = self.make_discriminator()

        generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

        seed = tf.random.normal([self.num_examples_to_generate, self.noise_dim])
        
        logging.info("Starting training process...")

        for epoch in range(epochs):
            start = time.time()
            
            epoch_gen_loss = []
            epoch_disc_loss = []

            for batch in dataset:
                gen_loss, disc_loss = self.train_step(batch, generator, discriminator,
                                                    generator_optimizer, discriminator_optimizer)
                epoch_gen_loss.append(float(gen_loss))
                epoch_disc_loss.append(float(disc_loss))

            # Generate and save images
            generated_samples = self.generate_and_save_images(generator, epoch + 1, seed)
            
            # Calculate quality metrics
            quality_metrics = self.calculate_quality_metrics(generated_samples, dataset[:16])
            
            # Get system metrics
            system_metrics = self.get_system_metrics()
            
            # Store metrics
            self.metrics['generator_loss'].append(np.mean(epoch_gen_loss))
            self.metrics['discriminator_loss'].append(np.mean(epoch_disc_loss))
            self.metrics['training_time'].append(time.time() - start)
            self.metrics['memory_usage'].append(system_metrics['memory_used_gb'])
            self.metrics['quality_metrics'].append(quality_metrics)

            logging.info(
                f'Epoch {epoch + 1}, Gen Loss: {np.mean(epoch_gen_loss):.4f}, '
                f'Disc Loss: {np.mean(epoch_disc_loss):.4f}, '
                f'Time: {time.time() - start:.2f} sec'
            )

        # Save final metrics
        self.save_metrics()
        return generator, discriminator

    def save_metrics(self):
        metrics_dir = 'training_metrics'
        if not os.path.exists(metrics_dir):
            os.makedirs(metrics_dir)
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{metrics_dir}/metrics_{timestamp}.json'
        
        with open(filename, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        
        logging.info(f"Metrics saved to {filename}")

    def plot_metrics(self):
        plt.figure(figsize=(15, 10))
        
        # Plot losses
        plt.subplot(2, 2, 1)
        plt.plot(self.metrics['generator_loss'], label='Generator')
        plt.plot(self.metrics['discriminator_loss'], label='Discriminator')
        plt.title('Training Losses')
        plt.legend()
        
        # Plot training time
        plt.subplot(2, 2, 2)
        plt.plot(self.metrics['training_time'])
        plt.title('Training Time per Epoch')
        plt.ylabel('Seconds')
        
        # Plot memory usage
        plt.subplot(2, 2, 3)
        plt.plot(self.metrics['memory_usage'])
        plt.title('Memory Usage')
        plt.ylabel('GB')
        
        # Plot quality metrics
        plt.subplot(2, 2, 4)
        quality_diff = [m['distribution_difference'] for m in self.metrics['quality_metrics']]
        plt.plot(quality_diff)
        plt.title('Generation Quality')
        plt.ylabel('Distribution Difference')
        
        plt.tight_layout()
        plt.savefig('training_metrics.png')
        plt.close()

def main():
    logging.info("Initializing GAN experiment...")
    
    # Create output directories
    for dir_name in ['generated_images', 'training_metrics', 'models']:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
    
    # Initialize GAN
    gan = IcebergGAN()
    
    try:
        # Load and prepare data
        # Note: You'll need to uncomment and modify this based on your data loading method
        # train_data = pd.read_json('train.json')
        # dataset = gan.load_and_prepare_data(train_data)
        
        # Train GAN
        generator, discriminator = gan.train(dataset, epochs=50)
        
        # Save models
        generator.save('models/generator.h5')
        discriminator.save('models/discriminator.h5')
        
        # Plot and save metrics
        gan.plot_metrics()
        
        logging.info("GAN experiment completed successfully")
        
    except Exception as e:
        logging.error(f"Error during GAN experiment: {str(e)}")
        raise

if __name__ == "__main__":
    main()
