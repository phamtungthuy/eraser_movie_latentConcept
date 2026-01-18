import argparse
import torch

import NeuroX.neurox.data.extraction.transformers_extractor as transformers_extractor

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_desc', type=str, default="bert-base-cased")
    parser.add_argument("--decompose_layers", action="store_true",
                        help="Save activations from each layer in a separate file")
    parser.add_argument("--include_special_tokens", action="store_true",
                        help="Include special tokens like [CLS] and [SEP] in the extracted representations")
    parser.add_argument("--filter_layers", default=None, type=str,
                        help="Comma separated list of layers to save activations for. The layers will be saved in the order specified in this argument.", )
    parser.add_argument("--input_type", choices=["text", "tsv"], 
                        help="Format of the input file, use tsv for multi-sentence inputs", default="text",)
    parser.add_argument('--input_corpus', type=str, default="/glue_ver/data/sst2_train.json")
    parser.add_argument('--output_file', type=str, default="tok.sent_len")
    parser.add_argument('--output_type', type=str, default="json")
    parser.add_argument('--device', type=str, default=None,
                        help="Device to use: 'cuda' or 'cpu'. Default: auto-detect (cuda if available)")

    args = parser.parse_args()

    # Auto-detect device if not specified
    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            print("Using CPU (no GPU detected)")
    else:
        device = args.device
        print(f"Using device: {device}")

    transformers_extractor.extract_representations(model_desc=args.model_desc,
                                                    input_corpus=args.input_corpus,
                                                    output_file=args.output_file,
                                                    output_type=args.output_type,
                                                    device=device,
                                                    decompose_layers=args.decompose_layers,
                                                    filter_layers=args.filter_layers,
                                                    include_special_tokens=args.include_special_tokens,
                                                    input_type=args.input_type)


if __name__ == "__main__":
    main()

