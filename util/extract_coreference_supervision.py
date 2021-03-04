"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the n2nmn project which
notice below and in LICENSE.n2nmn in the root directory of
this source tree.

Copyright (c) 2017, Ronghang Hu
All rights reserved.

Extracts coreference supervision for visdial dataset using off-the-shelf system.
"""

import argparse
import json
import sys
import neuralcoref
import spacy
from tqdm import tqdm as progressbar


def get_question_answer(data, i_dialog, i_question):
    """Extracts question + answer for a dialog turn.

    Args:
        data: Visdial data
        i_dialog: Index for the dialog
        i_question: Index for the turn
    """
    dialog = data["data"]["dialogs"][i_dialog]["dialog"][i_question]
    return (
        data["data"]["questions"][dialog["question"]],
        data["data"]["answers"][dialog["answer"]],
    )


def get_coref_cluster_sentence(utterance_cluster):
    """Visualize the co-reference clusters as string using, e.g [1   ].

    Args:
        utterance_cluster: Cluster corresponding to the utterance
    """
    sentence = ""
    for utterance in utterance_cluster:
        if not sentence:
            # print(utterance["sentence"])
            sentence = list(" " * (len(utterance["sentence"]) + 2))
        s = utterance["start_char"]
        sentence[s] = "["
        sentence[utterance["end_char"]] = "]"
        s += 1
        if not sentence[s] == " ":
            s += 2

        id_str = str(utterance["cluster_id"])
        sentence[s : (s + len(id_str))] = id_str

    # print("".join(sentence))
    return "".join(sentence)


def get_coref_cluster_list(utterance_cluster_map, ui):
    if ui in utterance_cluster_map:
        return utterance_cluster_map[ui]
    else:
        return []


def extract_corefs(data_file_name, out_file_name):
    print("Reading: {}".format(data_file_name))
    with open(data_file_name) as data_file:
        data = json.load(data_file)

    n_dialogs = len(data["data"]["dialogs"])
    coref = neuralcoref.Coref()
    # NOTE: neuralcoref gets stuck if there are numbers with an apostrophe.
    # Replacing them with equally long strings as a temporary fix.
    def remove_numbered_age(string):
        REPLACE_STRINGS = {
            "10's": "10ss",
            "20's": "20ss",
            "30's": "30ss",
            "40's": "40ss",
            "50's": "50ss",
            "60's": "60ss",
            "70's": "70ss",
            "80's": "80ss",
            "90's": "90ss",
            "100's": "100ss",
        }
        final_string = string
        for key, replacement in REPLACE_STRINGS.items():
            final_string = final_string.replace(key, replacement)
        return final_string

    for i_dialog in progressbar(range(n_dialogs)):
        dialog = data["data"]["dialogs"][i_dialog]
        str_dialog = dialog["caption"] + ". "
        list_dialog = [dialog["caption"] + "."]
        for i_question in range(len(dialog["dialog"])):
            q, a = get_question_answer(data, i_dialog, i_question)

            str_dialog += q + "? " + a + ". "
            list_dialog.append(q + "?")
            list_dialog.append(a + ".")

        list_dialog = [remove_numbered_age(ii) for ii in list_dialog]
        clusters = coref.one_shot_coref(utterances=list_dialog)
        mentions = coref.get_mentions()

        cluster_keys = list(clusters.keys())
        # match from utterance to cluster
        utterance_cluster_map = {}
        utterance_referrer_map = {}
        utterance_reference_map = {}
        for i_key in range(len(cluster_keys)):
            # assume reference is the first occurance
            reference = min(clusters[cluster_keys[i_key]])
            cluster_dict_ref = {}
            cluster_dict_ref["reference_sentence_id"] = mentions[
                reference
            ].utterance_index
            cluster_dict_ref["reference_start_word"] = mentions[reference].start
            cluster_dict_ref["reference_end_word"] = mentions[reference].end
            cluster_dict_ref["reference_start_char"] = mentions[reference].start_char
            cluster_dict_ref["reference_end_char"] = mentions[reference].end_char
            for i_mention in clusters[cluster_keys[i_key]]:
                cluster_dict = {}
                ui = mentions[i_mention].utterance_index
                cluster_dict["cluster_id"] = i_key
                cluster_dict["start_word"] = mentions[i_mention].start
                cluster_dict["end_word"] = mentions[i_mention].end
                cluster_dict["start_char"] = mentions[i_mention].start_char
                cluster_dict["end_char"] = mentions[i_mention].end_char
                cluster_dict["sentence"] = list_dialog[ui]
                if ui not in utterance_cluster_map:
                    utterance_cluster_map[ui] = []
                    utterance_referrer_map[ui] = []
                    utterance_reference_map[ui] = []
                utterance_cluster_map[ui].append(cluster_dict)
                if i_mention == reference:
                    utterance_reference_map[ui].append(cluster_dict)
                else:
                    cluster_dict.update(cluster_dict_ref)
                    utterance_referrer_map[ui].append(cluster_dict)

        cluster_list = get_coref_cluster_list(utterance_cluster_map, 0)
        data["data"]["dialogs"][i_dialog]["caption_coref_clusters"] = cluster_list
        data["data"]["dialogs"][i_dialog][
            "caption_coref_visualized"
        ] = get_coref_cluster_sentence(cluster_list)
        data["data"]["dialogs"][i_dialog][
            "caption_reference_clusters"
        ] = get_coref_cluster_list(utterance_reference_map, 0)

        for i_question in range(len(dialog["dialog"])):
            # set which utterance it came from
            cluster_list = get_coref_cluster_list(
                utterance_cluster_map, (i_question + 1) * 2
            )
            data["data"]["dialogs"][i_dialog]["dialog"][i_question][
                "answer_coref_clusters"
            ] = cluster_list
            data["data"]["dialogs"][i_dialog]["dialog"][i_question][
                "answer_coref_visualized"
            ] = get_coref_cluster_sentence(cluster_list)

            cluster_list = get_coref_cluster_list(
                utterance_cluster_map, (i_question) * 2 + 1
            )
            data["data"]["dialogs"][i_dialog]["dialog"][i_question][
                "question_coref_clusters"
            ] = cluster_list
            data["data"]["dialogs"][i_dialog]["dialog"][i_question][
                "question_coref_visualized"
            ] = get_coref_cluster_sentence(cluster_list)

            data["data"]["dialogs"][i_dialog]["dialog"][i_question][
                "answer_referrer_clusters"
            ] = get_coref_cluster_list(utterance_referrer_map, (i_question + 1) * 2)
            data["data"]["dialogs"][i_dialog]["dialog"][i_question][
                "question_referrer_clusters"
            ] = get_coref_cluster_list(utterance_referrer_map, (i_question) * 2 + 1)
            data["data"]["dialogs"][i_dialog]["dialog"][i_question][
                "answer_reference_clusters"
            ] = get_coref_cluster_list(utterance_reference_map, (i_question + 1) * 2)
            data["data"]["dialogs"][i_dialog]["dialog"][i_question][
                "question_reference_clusters"
            ] = get_coref_cluster_list(utterance_reference_map, (i_question) * 2 + 1)

    print("Saving: {}".format(out_file_name))
    with open(out_file_name, "w") as outfile:
        json.dump(data, outfile)
    return clusters, coref, data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_data_path", required=True, help="Path to VisDial JSON files"
    )
    parser.add_argument(
        "--output_save_path", default="-", help="Path to save the coreferences"
    )
    try:
        parsed_args = vars(parser.parse_args())
    except (IOError) as msg:
        parser.error(str(msg))
    extract_corefs(parsed_args["input_data_path"], parsed_args["output_save_path"])
