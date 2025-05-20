import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Conv2D, Concatenate, MaxPooling2D, GlobalAveragePooling2D,
    Activation, Multiply, Dense, GlobalMaxPooling2D, Dropout
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2  # Added for frame processing

def channel_attention_module(x, ratio=8):
    """
    Channel Attention Module for CBAM
    """
    # Get the shape of the input tensor
    batch, _, _, channel = x.shape
    
    # Define the fully connected layers to be shared between avg and max pooling
    l1 = Dense(channel // ratio, activation="relu", use_bias=False)
    l2 = Dense(channel, use_bias=False)
    
    # Average pooling branch
    x1 = GlobalAveragePooling2D()(x)
    x1 = l1(x1)
    x1 = l2(x1)
    
    # Max pooling branch
    x2 = GlobalMaxPooling2D()(x)
    x2 = l1(x2)
    x2 = l2(x2)
    
    # Combine the features and apply sigmoid activation
    feats = x1 + x2
    feats = Activation("sigmoid")(feats)
    
    # Apply channel attention
    feats = Multiply()([x, feats])
    return feats

def cbam_block(x):
    """
    Convolutional Block Attention Module (CBAM)
    Currently only implementing channel attention
    """
    x = channel_attention_module(x)
    return x

def aspp_block(input_layer, dilation_rates=[1, 2], num_filters=64):
    """
    Atrous Spatial Pyramid Pooling block with different dilation rates
    """
    aspp_layers = []
    for dilation_rate in dilation_rates:
        aspp_layer = Conv2D(
            num_filters, 
            (3, 3), 
            padding='same', 
            activation='relu', 
            dilation_rate=dilation_rate
        )(input_layer)
        aspp_layers.append(aspp_layer)
    
    # Concatenate the layers with different dilation rates
    concatenated = Concatenate()(aspp_layers)
    return concatenated

def create_aspp_model(input_shape):
    """
    Create the ASPP model with CBAM attention
    """
    input_layer = Input(shape=input_shape)
    
    # First ASPP block
    aspp_block1 = aspp_block(input_layer, dilation_rates=[1, 5], num_filters=64)
    aspp_block1 = cbam_block(aspp_block1)
    maxpool1 = MaxPooling2D(pool_size=(2, 2))(aspp_block1)
    
    # Second ASPP block
    aspp_block2 = aspp_block(maxpool1, dilation_rates=[1, 3], num_filters=64)
    aspp_block2 = cbam_block(aspp_block2)
    maxpool2 = MaxPooling2D(pool_size=(2, 2))(aspp_block2)
    
    # Create model
    aspp_model = tf.keras.Model(inputs=input_layer, outputs=maxpool2)
    return aspp_model

def create_hybrid_model(input_shape, swin_model_path='Code\\SwinModel', num_classes=30):
    """
    Create a hybrid model that combines ASPP and Swin Transformer
    """
    # ASPP CNN branch
    aspp_model = create_aspp_model(input_shape)
    x = tf.keras.layers.BatchNormalization()(aspp_model.output)
    x = GlobalAveragePooling2D()(x)
    
    # Swin Transformer branch
    input_swin = Input(shape=input_shape)
    
    # Load Swin Transformer model
    print(f"Loading Swin Transformer model from: {swin_model_path}")
    try:
        swin = load_model(swin_model_path)
    except Exception as e:
        print(f"Error loading Swin model: {e}")
        print("Trying to load with absolute path...")
        # Try with absolute path if relative path fails
        script_dir = os.path.dirname(os.path.abspath(__file__))
        abs_path = os.path.join(script_dir, swin_model_path)
        print(f"Trying path: {abs_path}")
        swin = load_model(abs_path)
    
    # Freeze Swin Transformer layers
    for layer in swin.layers:
        layer.trainable = False
    
    swin_features = swin(input_swin, training=False)
    
    # Process ASPP features
    x = Dropout(0.3)(x)
    x = Dense(256, activation='selu')(x)
    x = Dropout(0.3)(x)
    x_cnn = Dense(num_classes, activation='selu')(x)
    
    # Combine features from both models
    combined_features = Concatenate()([x_cnn, swin_features])
    
    # Final classification layers
    x = Dense(128, activation='selu')(combined_features)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='selu')(x)
    x = Dropout(0.3)(x)
    x = Dense(num_classes, activation='selu')(x)
    output = Dense(num_classes, activation='softmax')(x)
    
    # Create the complete model
    model = Model(inputs=[aspp_model.input, input_swin], outputs=output)
    return model

def generator_two_img(gen):
    """
    Custom generator to feed the same image to both inputs
    """
    while True:
        X1i = next(gen)
        yield [X1i[0], X1i[0]], X1i[1]

def train_model(model, train_data_dir, validation_data_dir, output_model_path, 
               batch_size=32, epochs=50, img_size=224):
    """
    Train the hybrid model with data generators
    """
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    
    # Only rescaling for validation
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    validation_generator = val_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    # Compile the model
    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.experimental.Adamax(),
        metrics=['accuracy']
    )
    
    # Train the model
    history = model.fit(
        generator_two_img(train_generator),
        validation_data=generator_two_img(validation_generator),
        steps_per_epoch=train_generator.samples // batch_size,
        validation_steps=validation_generator.samples // batch_size,
        epochs=epochs
    )
    
    # Save the model without optimizer state to reduce size
    model.save(output_model_path, include_optimizer=False)
    
    return history, model

def process_frames(model, frames=None, video_path=None, save_output=None, img_size=224):
    """
    Process frames through the model
    
    Args:
        model: Loaded model
        frames: List of frames (numpy arrays), optional
        video_path: Path to video file to process, optional
        save_output: Path to save processed video with predictions, optional
        img_size: Size to resize frames to
        
    Returns:
        Predictions for each frame and processed frames
    """
    processed_frames = []
    predictions = []
    
    # If video path is provided, get frames from the video
    if video_path and not frames:
        frames = []
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        print(f"Extracted {len(frames)} frames from video")
    
    # If no frames are provided, return empty results
    if not frames:
        print("No frames provided and no video path given")
        return [], []
    
    # Process each frame
    for frame in frames:
        # Convert BGR to RGB (OpenCV uses BGR by default)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize frame if needed
        if frame_rgb.shape[0] != img_size or frame_rgb.shape[1] != img_size:
            frame_resized = cv2.resize(frame_rgb, (img_size, img_size))
        else:
            frame_resized = frame_rgb
        
        # Normalize pixel values
        frame_normalized = frame_resized / 255.0
        
        processed_frames.append(frame_normalized)
    
    # Convert to numpy array
    processed_frames_array = np.array(processed_frames)
    
    # Predict using the model (same input for both branches)
    try:
        batch_predictions = model.predict([processed_frames_array, processed_frames_array])
        predictions = batch_predictions
    except Exception as e:
        print(f"Error during prediction: {e}")
        return [], processed_frames
    
    # Save output video if requested
    if save_output and len(frames) > 0:
        # Get frame dimensions
        height, width = frames[0].shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(save_output, fourcc, 30.0, (width, height))
        
        # Get class with highest probability for each frame
        pred_classes = np.argmax(predictions, axis=1)
        
        # Process and write each frame
        for i, frame in enumerate(frames):
            # Add prediction text to frame
            pred_class = pred_classes[i]
            confidence = predictions[i][pred_class]
            text = f"Class: {pred_class}, Conf: {confidence:.2f}"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Write frame to output video
            out.write(frame)
        
        out.release()
        print(f"Processed video saved to {save_output}")
    
    return predictions, processed_frames

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train or use ASPP-Swin hybrid model')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'process'],
                        help='Mode: train the model or process frames')
    parser.add_argument('--swin_model_path', type=str, default='Code\\SwinModel',
                        help='Path to the pre-trained Swin Transformer model')
    parser.add_argument('--output_model_path', type=str, required=True,
                        help='Path to save the trained model')
    parser.add_argument('--train_data_dir', type=str, help='Directory with training data')
    parser.add_argument('--validation_data_dir', type=str, help='Directory with validation data')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training')
    parser.add_argument('--img_size', type=int, default=224, help='Image size for model input')
    parser.add_argument('--num_classes', type=int, default=30, help='Number of output classes')
    parser.add_argument('--video_path', type=str, help='Path to video file to process')
    parser.add_argument('--save_output', type=str, help='Path to save processed video')
    parser.add_argument('--camera', action='store_true', help='Use camera for live processing')
    
    args = parser.parse_args()
    
    # Define input shape
    input_shape = (args.img_size, args.img_size, 3)
    
    if args.mode == 'train':
        # Check if required arguments are provided
        if not args.train_data_dir or not args.validation_data_dir:
            parser.error("Training mode requires --train_data_dir and --validation_data_dir")
        
        # Create and train the model
        model = create_hybrid_model(input_shape, args.swin_model_path, args.num_classes)
        history, model = train_model(
            model, 
            args.train_data_dir, 
            args.validation_data_dir, 
            args.output_model_path,
            args.batch_size,
            args.epochs,
            args.img_size
        )
        print(f"Model training completed and saved to {args.output_model_path}")
        
    elif args.mode == 'process':
        # Make sure the output model path is provided
        if not os.path.exists(args.output_model_path):
            # If output_model_path doesn't exist, check if we need to create a new model
            print(f"Model not found at {args.output_model_path}")
            print("Creating a new model...")
            model = create_hybrid_model(input_shape, args.swin_model_path, args.num_classes)
            # Save the untrained model
            model.compile(
                loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.experimental.Adamax(),
                metrics=['accuracy']
            )
            model.save(args.output_model_path, include_optimizer=False)
            print(f"Created and saved new model to {args.output_model_path}")
        else:
            # Load the model for processing frames
            print(f"Loading model from {args.output_model_path}")
            try:
                model = load_model(args.output_model_path)
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Creating a new model instead...")
                model = create_hybrid_model(input_shape, args.swin_model_path, args.num_classes)
                model.compile(
                    loss="categorical_crossentropy",
                    optimizer=tf.keras.optimizers.experimental.Adamax(),
                    metrics=['accuracy']
                )
        
        print("Model loaded successfully. Use the model.predict() function to process frames.")
        
        # Process video if specified
        if args.video_path:
            print(f"Processing video: {args.video_path}")
            predictions, _ = process_frames(
                model, 
                video_path=args.video_path, 
                save_output=args.save_output, 
                img_size=args.img_size
            )
            print(f"Video processing complete. Made {len(predictions)} predictions.")
        
        # Use camera for live processing if specified
        elif args.camera:
            print("Starting camera for live processing...")
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                print("Error: Could not open camera.")
                return
            
            # Set up video writer if save_output is specified
            out = None
            if args.save_output:
                ret, frame = cap.read()
                if ret:
                    height, width = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    out = cv2.VideoWriter(args.save_output, fourcc, 20.0, (width, height))
            
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("Failed to grab frame")
                        break
                    
                    # Process the frame
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_resized = cv2.resize(frame_rgb, (args.img_size, args.img_size))
                    frame_normalized = frame_resized / 255.0
                    
                    # Make prediction
                    prediction = model.predict([np.array([frame_normalized]), np.array([frame_normalized])])
                    pred_class = np.argmax(prediction[0])
                    confidence = prediction[0][pred_class]
                    
                    # Add prediction text to frame
                    text = f"Class: {pred_class}, Conf: {confidence:.2f}"
                    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Display the frame
                    cv2.imshow('Frame', frame)
                    
                    # Write frame if saving
                    if out is not None:
                        out.write(frame)
                    
                    # Break loop on 'q' key press
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
            finally:
                cap.release()
                if out is not None:
                    out.release()
                cv2.destroyAllWindows()
                print("Camera processing stopped")
        
        else:
            print("No video or camera specified. Ready for frame processing.")
            print("Example usage:")
            print("  python aspp_swin_model.py --mode process --output_model_path model.h5 --video_path input.mp4 --save_output output.avi")
            print("  python aspp_swin_model.py --mode process --output_model_path model.h5 --camera --save_output output.avi")

if __name__ == "__main__":
    main()
