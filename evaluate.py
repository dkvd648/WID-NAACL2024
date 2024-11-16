import argparse
import torch
from uer.model_loader import load_model
from uer.models.bert_classifier import BertClassifier

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the test dataset.")
    parser.add_argument("--pretrained_model_path", type=str, required=True, help="Path to the fine-tuned model.")
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to the vocab file.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the model config.")
    parser.add_argument("--target", type=str, required=True, help="Task type (e.g., classification).")
    parser.add_argument("--labels_num", type=int, required=True, help="Number of labels.")

    args = parser.parse_args()

    # Load the model
    print("Loading model...")
    model = BertClassifier(args.config_path, args.vocab_path, target=args.target, labels_num=args.labels_num)
    model = load_model(model, args.pretrained_model_path)

    # Load dataset
    print("Loading dataset...")
    dataset = torch.load(args.dataset_path)

    # Evaluate the model
    print("Evaluating...")
    model.eval()
    correct, total = 0, 0
    for data in dataset:
        input_tensor, labels = data
        with torch.no_grad():
            output = model(input_tensor)
            predictions = torch.argmax(output, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
