from guidance import models, gen
from guidance import select
import json
from datasets import concatenate_datasets, load_dataset, Dataset
from transformers import AutoTokenizer
from tokenizers import pre_tokenizers 
import guidance 
from huggingface_hub import login

path = "meta-llama/Meta-Llama-3-8B"

tokenizer = AutoTokenizer.from_pretrained(
    path,
    padding_side="left",
    trust_remote_code=True,  # token=hf_token
)
byte_decoder = {}
alphabet = pre_tokenizers.ByteLevel(False, False).alphabet()
known_vals = set([])
for j in range(256):
    for k in range(256):
        for l in range(256):
            if len(byte_decoder.keys()) < 256:
                b = b""
                vals = [j,k,l]
                if not set(vals).issubset(known_vals):
                    for d in range(3):
                        b = b + int.to_bytes(vals[d])
                    try:
                        c = b.decode()
                        t = pre_tokenizers.ByteLevel(False,False).pre_tokenize_str(c)[0][0]
                        for m in range(3):
                            if t[m] not in byte_decoder.keys():
                                byte_decoder[t[m]] = vals[m]
                                known_vals.add(vals[m])
                    except UnicodeDecodeError:
                        pass

byte_decoder['À'] = 192
byte_decoder['Á'] = 193
byte_decoder['ð'] = 240
byte_decoder['ñ'] = 241
byte_decoder['ò'] = 242
byte_decoder['ó'] = 243
byte_decoder['ô'] = 244
byte_decoder['õ'] = 245
byte_decoder['ö'] = 246
byte_decoder['÷'] = 247
byte_decoder['ø'] = 248
byte_decoder['ù'] = 249
byte_decoder['ú'] = 250
byte_decoder['û'] = 251
byte_decoder['ü'] = 252
byte_decoder['ý'] = 253
byte_decoder['þ'] = 254
byte_decoder['ÿ'] = 255
tokenizer.byte_decoder = byte_decoder

llama3 = models.Transformers(path, tokenizer=tokenizer)

dev_path = "data/dev_correction.jsonl"
with open(dev_path, "r") as f:
    disambiguation_dev_data = [json.loads(line) for line in f]
def gen_dev():
    for dev_id, dev_item in enumerate(disambiguation_dev_data): 
        yield {"source":dev_item["source"], "evidence": dev_item["evidence"], "target":dev_item["target"][0]}
dis_dev_dataset = Dataset.from_generator(gen_dev)

test_path = "data/test_correction.jsonl"
with open(test_path, "r") as f:
    disambiguation_test_data = [json.loads(line) for line in f]
def gen_test():
    for test_id, test_item in enumerate(disambiguation_test_data): 
        yield {"source":test_item["source"], "evidence": test_item["evidence"], "target":test_item["target"][0]}
dis_test_dataset = Dataset.from_generator(gen_test)

@guidance
def disambiguation_eight_shot(lm, question):
    lm += '''\
    Please make the following claim less ambiguous with regard to the following evidence, as in the examples below. 

    Claim: bridges of madison county is a true story.
    Evidence: The Bridges of Madison County (also published as Love in Black and White) is a 1992 best-selling romance novella by American writer Robert James Waller that tells the story of a married Italian-American woman (WW2 \u2018War bride\u2019) living on a Madison County, Iowa, farm in the 1960s. While her husband and children are away at the State Fair, she engages in an affair with a National Geographic photographer from Bellingham, Washington, who is visiting Madison County to create a photographic essay on the covered bridges in the area. The novel is presented as a novelization of a true story, but it is in fact entirely fictional. The novel is one of the bestselling books of the 20th century, with 60 million copies sold world-wide. It has also been adapted into a feature film in 1995 and a musical in 2013.
    Revised claim: bridges of madison county is a fictional story

    Claim: you can keep a gray wolf as a pet.
    Evidence: Some wildlife centers housing captive wolves prohibit handlers from entering wolf enclosures if they happen to have a cold or other vulnerability which the wolves can detect. Captive wolves are generally shy and avoid eye contact with humans other than their owner, as well as not listening to any commands made by any other humans. They usually vacate rooms or hide when a new person enters the establishment. Even seemingly friendly wolves need to be treated with caution, as captive wolves tend to view and treat people as other wolves, and will thus bite or dominate people in the same situation in which they would other wolves. Ordinary pet food is inadequate, as an adult wolf needs 1\u20132.5 kg (2\u20135 lbs) of meat daily along with bones, skin and fur to meet its nutritional requirements. Wolves may defend their food against people, and react violently to people trying to remove it. The exercise needs of a wolf exceed the average dog's demand. Because of this, captive wolves typically do not cope well in urban areas. Due to their talent at observational learning, adult captive wolves can quickly work out how to escape confinement, and require constant reinforcement by caretakers or owners, which makes raising wolves difficult for people who raise their pets in an even, rather than subordinate, environment.
    Revised claim: it is difficult to raise a wolf as a pet.

    Claim: it is illegal to flash your headlights to warn off the police in the uk.
    Evidence: Though not all of its rules represent law, the Highway Code states \"Only flash your headlights to let other road users know that you are there. Do not flash your headlights in an attempt to intimidate other road users\". Drivers warning others about speed traps have been fined in the past for \"misuse of headlights\".   Headlight flashing in the United Kingdom is often used as a signal that the driver flashing you is offering to let you go first. Such use is however strongly discouraged because it can lead to accidents where the driver flashing has not seen the approach of another road user. Using it to indicate that you are coming through and the other driver must wait, could lead to an accident.   Drivers should also be aware of the so-called \"Flash-for-Cash\" scam, in which criminals flash their lights to let other drivers out of a junction, then crash into them on purpose in order to make fraudulent insurance claims for damage and whiplash injury.
    Revised claim: In the UK, you should only flash your headlights to let other drivers know you are there.

    Claim: you do need intent to commit a crime
    Evidence: In criminal law, intent is a subjective state of mind that must accompany the acts of certain crimes to constitute a violation. A more formal, generally synonymous legal term is scienter: intent or knowledge of wrongdoing.
    Revised claim: you do need intent to commit some crimes

    Claim: running with scissors is based on a true story.
    Evidence: In 2005, the family of Dr. Rodolph H. Turcotte (1919\u20132000), of Massachusetts filed suit against Burroughs and his publisher, alleging defamation of character and invasion of privacy. They stated that they were the basis for the Finch family portrayed in the book but that Burroughs had fabricated or exaggerated various descriptions of their activities.   > It's still a memoir, it's marketed as a memoir, they've agreed one hundred percent that it is a memoir.   The case was later settled with Sony Pictures Entertainment in October 2006, prior to the release of the film adaptation. Burroughs and his publisher, St. Martin's Press, settled with the Turcotte family in August 2007. The Turcottes were reportedly seeking damages of $2 million for invasion of privacy, defamation, and emotional distress; the Turcottes alleged Running with Scissors was largely fictional and written in a sensational manner. Burroughs defended his work as \"entirely accurate\", but agreed to call the work a \"book\" (instead of a \"memoir\") in the author's note, to alter the acknowledgments page in future editions to recognize the Turcotte family's conflicting memories of described events, and express regret for \"any unintentional harm\" to the Turcotte family. Burroughs felt vindicated by the settlement. \"I'm not at all sorry that I wrote [the book]. And you know, the suit settled\u2014it settled in my favor. I didn't change a word of the memoir, not one word of it. It's still a memoir, it's marketed as a memoir, they've agreed one hundred percent that it is a memoir\".   Future printings of Running with Scissors will contain modified language in the Author's Note and Acknowledgments pages. Where the Acknowledgments page had read: \"Additionally, I would like to thank each and every member of a certain Massachusetts family for taking me into their home and accepting me as one of their own,\" the following was substituted: \"Additionally, I would like to thank the real-life members of the family portrayed in this book for taking me into their home and accepting me as one of their own. I recognize that their memories of the events described in this book are different than my own. They are each fine, decent, and hard-working people. The book was not intended to hurt the family. Both my publisher and I regret any unintentional harm resulting from the publishing and marketing of Running with Scissors\"
    Revised claim: running with scissors is somewhat based on the recollections of part of the author's life

    Claim: the united states does recognize kosovo as a country.
    Evidence: A number of states expressed concern over the unilateral character of Kosovo's declaration, or explicitly announced that they would not recognise an independent Kosovo. The United Nations Security Council (UNSC) remains divided on this issue: of its five members with veto power, three (the United States, the United Kingdom, and France) have recognised the declaration of independence, while the People's Republic of China has expressed concern, urging the continuation of the previous negotiation framework. The Russian Federation has rejected the declaration and considers it illegal. In May 2008, Russia, China, and India released a joint statement calling for new negotiations between Belgrade and Pristina.   Although EU member states individually decide whether to recognise Kosovo, by consensus the EU has commissioned the European Union Rule of Law Mission in Kosovo (EULEX) to ensure peace and continued external oversight. Due to the dispute in the United Nations Security Council (UNSC), the reconfiguration of the United Nations Interim Administration Mission in Kosovo (UNMIK) and partial handover to the EULEX mission met with difficulties. In spite of Russian and Serbian protests, the UN Secretary-General Ban Ki-moon proceeded with the reconfiguration plan. On 15 July 2008, he stated: \"In the light of the fact that the Security Council is unable to provide guidance, I have instructed my Special Representative to move forward with the reconfiguration of UNMIK ... in order to adapt UNMIK to a changed reality\". According to the Secretary-General, the \"United Nations has maintained a position of strict neutrality on the question of Kosovo's status\". On 26 November 2008, the UNSC gave the green light to the deployment of the EULEX mission in Kosovo. The EU mission is to assume police, justice, and customs duties from the UN, while operating under the United Nations Security Council Resolution 1244 (UNSCR 1244) that first placed Kosovo under UN administration in 1999.
    Revised claim: the united states does recognize kosovo as a country.

    Claim: the jeep compass does not come in a 6 cylinder.
    Evidence: The second-generation of Jeep Compass debuted on September 27, 2016 in Brazil and at the Los Angeles International Auto Show in November 2016, replacing the Jeep Patriot and first generation Compass. Production for North American-market Compass models was moved to Toluca, Mexico, while Jeep Cherokee (KL) production will move from Toledo, Ohio to Belvidere, Illinois, where the first-generation Compass and Jeep Patriot were both assembled.   Using a stretched version of the same platform as the Renegade, the Compass is available in four distinct trim levels: the base Sport, the mid-level Latitude, the luxurious Limited, and the off road-capable Trailhawk. All trim levels are available with either front-wheel drive or four-wheel drive, with the exception of the Trailhawk, which is only available in a 4WD configuration.   In the United States, the Compass comes equipped with a 2.4L Tigershark four-cylinder engine or a 1.4L MultiAir four-cylinder engine. More than 65 percent of the upper body structure and frame is made of high-strength steel.   Styling of the second-generation Compass is inspired by two of its larger siblings, the Jeep Cherokee (KL) and Jeep Grand Cherokee (WK2). Styling elements taken from the Grand Cherokee include headlamps integrated into the front grille and a narrow front grille with a black finish, while the Cherokee lends some of its rear end styling elements, basic interior design, and 2.4L Tigershark inline four-cylinder (I4) gasoline engine and ZF-sourced nine-speed 948TE automatic transmission to the overall design of the Compass.
    Revised claim: It is not clear from the evidence whether the jeep compass does not come in a 6 cylinder.

    Claim: a woodchuck and a groundhog are the same thing.
    Evidence: Both their diet and habit of burrowing make them serious nuisance animals around farms and gardens. They will eat many commonly grown vegetables, and their burrows can destroy farm ponds and undermine foundations.   Very often the dens of groundhogs provide homes for other animals including skunks, red foxes, and cottontail rabbits. The fox and skunk feed upon field mice, grasshoppers, beetles and other creatures that destroy farm crops. In aiding these animals, the groundhog indirectly helps the farmer. In addition to providing homes for itself and other animals, the groundhog aids in soil improvement by bringing subsoil to the surface. The groundhog is also a valuable game animal and is considered a difficult sport when hunted in a fair manner. In some parts of the U.S., they have been eaten.   A report in 1883 by the New Hampshire Legislative Woodchuck Committee humorously describes the groundhog's objectionable character:
    Revised claim: It is not clear from the evidence whether a woodchuck and a groundhog are the same thing.

    '''
    lm += f'Question: {question}\n'
    # Generate until end of sentence
    lm += 'Revised claim: ' + gen(stop="\n")
    return lm


@guidance
def disambiguation_four_shot(lm, question):
    lm += '''\
    Please make the following claim less ambiguous with regard to the following evidence, as in the examples below. 

    Claim: bridges of madison county is a true story.
    Evidence: The Bridges of Madison County (also published as Love in Black and White) is a 1992 best-selling romance novella by American writer Robert James Waller that tells the story of a married Italian-American woman (WW2 \u2018War bride\u2019) living on a Madison County, Iowa, farm in the 1960s. While her husband and children are away at the State Fair, she engages in an affair with a National Geographic photographer from Bellingham, Washington, who is visiting Madison County to create a photographic essay on the covered bridges in the area. The novel is presented as a novelization of a true story, but it is in fact entirely fictional. The novel is one of the bestselling books of the 20th century, with 60 million copies sold world-wide. It has also been adapted into a feature film in 1995 and a musical in 2013.
    Revised claim: bridges of madison county is a fictional story

    Claim: you can keep a gray wolf as a pet.
    Evidence: Some wildlife centers housing captive wolves prohibit handlers from entering wolf enclosures if they happen to have a cold or other vulnerability which the wolves can detect. Captive wolves are generally shy and avoid eye contact with humans other than their owner, as well as not listening to any commands made by any other humans. They usually vacate rooms or hide when a new person enters the establishment. Even seemingly friendly wolves need to be treated with caution, as captive wolves tend to view and treat people as other wolves, and will thus bite or dominate people in the same situation in which they would other wolves. Ordinary pet food is inadequate, as an adult wolf needs 1\u20132.5 kg (2\u20135 lbs) of meat daily along with bones, skin and fur to meet its nutritional requirements. Wolves may defend their food against people, and react violently to people trying to remove it. The exercise needs of a wolf exceed the average dog's demand. Because of this, captive wolves typically do not cope well in urban areas. Due to their talent at observational learning, adult captive wolves can quickly work out how to escape confinement, and require constant reinforcement by caretakers or owners, which makes raising wolves difficult for people who raise their pets in an even, rather than subordinate, environment.
    Revised claim: it is difficult to raise a wolf as a pet.

    Claim: the united states does recognize kosovo as a country.
    Evidence: A number of states expressed concern over the unilateral character of Kosovo's declaration, or explicitly announced that they would not recognise an independent Kosovo. The United Nations Security Council (UNSC) remains divided on this issue: of its five members with veto power, three (the United States, the United Kingdom, and France) have recognised the declaration of independence, while the People's Republic of China has expressed concern, urging the continuation of the previous negotiation framework. The Russian Federation has rejected the declaration and considers it illegal. In May 2008, Russia, China, and India released a joint statement calling for new negotiations between Belgrade and Pristina.   Although EU member states individually decide whether to recognise Kosovo, by consensus the EU has commissioned the European Union Rule of Law Mission in Kosovo (EULEX) to ensure peace and continued external oversight. Due to the dispute in the United Nations Security Council (UNSC), the reconfiguration of the United Nations Interim Administration Mission in Kosovo (UNMIK) and partial handover to the EULEX mission met with difficulties. In spite of Russian and Serbian protests, the UN Secretary-General Ban Ki-moon proceeded with the reconfiguration plan. On 15 July 2008, he stated: \"In the light of the fact that the Security Council is unable to provide guidance, I have instructed my Special Representative to move forward with the reconfiguration of UNMIK ... in order to adapt UNMIK to a changed reality\". According to the Secretary-General, the \"United Nations has maintained a position of strict neutrality on the question of Kosovo's status\". On 26 November 2008, the UNSC gave the green light to the deployment of the EULEX mission in Kosovo. The EU mission is to assume police, justice, and customs duties from the UN, while operating under the United Nations Security Council Resolution 1244 (UNSCR 1244) that first placed Kosovo under UN administration in 1999.
    Revised claim: the united states does recognize kosovo as a country.

    Claim: the jeep compass does not come in a 6 cylinder.
    Evidence: The second-generation of Jeep Compass debuted on September 27, 2016 in Brazil and at the Los Angeles International Auto Show in November 2016, replacing the Jeep Patriot and first generation Compass. Production for North American-market Compass models was moved to Toluca, Mexico, while Jeep Cherokee (KL) production will move from Toledo, Ohio to Belvidere, Illinois, where the first-generation Compass and Jeep Patriot were both assembled.   Using a stretched version of the same platform as the Renegade, the Compass is available in four distinct trim levels: the base Sport, the mid-level Latitude, the luxurious Limited, and the off road-capable Trailhawk. All trim levels are available with either front-wheel drive or four-wheel drive, with the exception of the Trailhawk, which is only available in a 4WD configuration.   In the United States, the Compass comes equipped with a 2.4L Tigershark four-cylinder engine or a 1.4L MultiAir four-cylinder engine. More than 65 percent of the upper body structure and frame is made of high-strength steel.   Styling of the second-generation Compass is inspired by two of its larger siblings, the Jeep Cherokee (KL) and Jeep Grand Cherokee (WK2). Styling elements taken from the Grand Cherokee include headlamps integrated into the front grille and a narrow front grille with a black finish, while the Cherokee lends some of its rear end styling elements, basic interior design, and 2.4L Tigershark inline four-cylinder (I4) gasoline engine and ZF-sourced nine-speed 948TE automatic transmission to the overall design of the Compass.
    Revised claim: It is not clear from the evidence whether the jeep compass does not come in a 6 cylinder.

    '''
    lm += f'Question: {question}\n'
    # Generate until end of sentence
    lm += 'Revised claim: ' + gen(stop="\n")
    return lm


@guidance
def disambiguation_ambiguous_four_shot(lm, question):
    lm += '''\
    Please make the following claim less ambiguous with regard to the following evidence, as in the examples below. 

    Claim: bridges of madison county is a true story.
    Evidence: The Bridges of Madison County (also published as Love in Black and White) is a 1992 best-selling romance novella by American writer Robert James Waller that tells the story of a married Italian-American woman (WW2 \u2018War bride\u2019) living on a Madison County, Iowa, farm in the 1960s. While her husband and children are away at the State Fair, she engages in an affair with a National Geographic photographer from Bellingham, Washington, who is visiting Madison County to create a photographic essay on the covered bridges in the area. The novel is presented as a novelization of a true story, but it is in fact entirely fictional. The novel is one of the bestselling books of the 20th century, with 60 million copies sold world-wide. It has also been adapted into a feature film in 1995 and a musical in 2013.
    Revised claim: bridges of madison county is a fictional story

    Claim: you can keep a gray wolf as a pet.
    Evidence: Some wildlife centers housing captive wolves prohibit handlers from entering wolf enclosures if they happen to have a cold or other vulnerability which the wolves can detect. Captive wolves are generally shy and avoid eye contact with humans other than their owner, as well as not listening to any commands made by any other humans. They usually vacate rooms or hide when a new person enters the establishment. Even seemingly friendly wolves need to be treated with caution, as captive wolves tend to view and treat people as other wolves, and will thus bite or dominate people in the same situation in which they would other wolves. Ordinary pet food is inadequate, as an adult wolf needs 1\u20132.5 kg (2\u20135 lbs) of meat daily along with bones, skin and fur to meet its nutritional requirements. Wolves may defend their food against people, and react violently to people trying to remove it. The exercise needs of a wolf exceed the average dog's demand. Because of this, captive wolves typically do not cope well in urban areas. Due to their talent at observational learning, adult captive wolves can quickly work out how to escape confinement, and require constant reinforcement by caretakers or owners, which makes raising wolves difficult for people who raise their pets in an even, rather than subordinate, environment.
    Revised claim: it is difficult to raise a wolf as a pet.

    Claim: it is illegal to flash your headlights to warn off the police in the uk.
    Evidence: Though not all of its rules represent law, the Highway Code states \"Only flash your headlights to let other road users know that you are there. Do not flash your headlights in an attempt to intimidate other road users\". Drivers warning others about speed traps have been fined in the past for \"misuse of headlights\".   Headlight flashing in the United Kingdom is often used as a signal that the driver flashing you is offering to let you go first. Such use is however strongly discouraged because it can lead to accidents where the driver flashing has not seen the approach of another road user. Using it to indicate that you are coming through and the other driver must wait, could lead to an accident.   Drivers should also be aware of the so-called \"Flash-for-Cash\" scam, in which criminals flash their lights to let other drivers out of a junction, then crash into them on purpose in order to make fraudulent insurance claims for damage and whiplash injury.
    Revised claim: In the UK, you should only flash your headlights to let other drivers know you are there.

    Claim: you do need intent to commit a crime
    Evidence: In criminal law, intent is a subjective state of mind that must accompany the acts of certain crimes to constitute a violation. A more formal, generally synonymous legal term is scienter: intent or knowledge of wrongdoing.
    Revised claim: you do need intent to commit some crimes

    '''
    lm += f'Question: {question}\n'
    # Generate until end of sentence
    lm += 'Revised claim: ' + gen(stop="\n")
    return lm

@guidance
def disambiguation_ambiguous_eight_shot(lm, question):
    lm += '''\
    Please make the following claim less ambiguous with regard to the following evidence, as in the examples below. 

    Claim: bridges of madison county is a true story.
    Evidence: The Bridges of Madison County (also published as Love in Black and White) is a 1992 best-selling romance novella by American writer Robert James Waller that tells the story of a married Italian-American woman (WW2 \u2018War bride\u2019) living on a Madison County, Iowa, farm in the 1960s. While her husband and children are away at the State Fair, she engages in an affair with a National Geographic photographer from Bellingham, Washington, who is visiting Madison County to create a photographic essay on the covered bridges in the area. The novel is presented as a novelization of a true story, but it is in fact entirely fictional. The novel is one of the bestselling books of the 20th century, with 60 million copies sold world-wide. It has also been adapted into a feature film in 1995 and a musical in 2013.
    Revised claim: bridges of madison county is a fictional story

    Claim: you can keep a gray wolf as a pet.
    Evidence: Some wildlife centers housing captive wolves prohibit handlers from entering wolf enclosures if they happen to have a cold or other vulnerability which the wolves can detect. Captive wolves are generally shy and avoid eye contact with humans other than their owner, as well as not listening to any commands made by any other humans. They usually vacate rooms or hide when a new person enters the establishment. Even seemingly friendly wolves need to be treated with caution, as captive wolves tend to view and treat people as other wolves, and will thus bite or dominate people in the same situation in which they would other wolves. Ordinary pet food is inadequate, as an adult wolf needs 1\u20132.5 kg (2\u20135 lbs) of meat daily along with bones, skin and fur to meet its nutritional requirements. Wolves may defend their food against people, and react violently to people trying to remove it. The exercise needs of a wolf exceed the average dog's demand. Because of this, captive wolves typically do not cope well in urban areas. Due to their talent at observational learning, adult captive wolves can quickly work out how to escape confinement, and require constant reinforcement by caretakers or owners, which makes raising wolves difficult for people who raise their pets in an even, rather than subordinate, environment.
    Revised claim: it is difficult to raise a wolf as a pet.

    Claim: it is illegal to flash your headlights to warn off the police in the uk.
    Evidence: Though not all of its rules represent law, the Highway Code states \"Only flash your headlights to let other road users know that you are there. Do not flash your headlights in an attempt to intimidate other road users\". Drivers warning others about speed traps have been fined in the past for \"misuse of headlights\".   Headlight flashing in the United Kingdom is often used as a signal that the driver flashing you is offering to let you go first. Such use is however strongly discouraged because it can lead to accidents where the driver flashing has not seen the approach of another road user. Using it to indicate that you are coming through and the other driver must wait, could lead to an accident.   Drivers should also be aware of the so-called \"Flash-for-Cash\" scam, in which criminals flash their lights to let other drivers out of a junction, then crash into them on purpose in order to make fraudulent insurance claims for damage and whiplash injury.
    Revised claim: In the UK, you should only flash your headlights to let other drivers know you are there.

    Claim: you do need intent to commit a crime
    Evidence: In criminal law, intent is a subjective state of mind that must accompany the acts of certain crimes to constitute a violation. A more formal, generally synonymous legal term is scienter: intent or knowledge of wrongdoing.
    Revised claim: you do need intent to commit some crimes

    Claim: running with scissors is based on a true story.
    Evidence: In 2005, the family of Dr. Rodolph H. Turcotte (1919\u20132000), of Massachusetts filed suit against Burroughs and his publisher, alleging defamation of character and invasion of privacy. They stated that they were the basis for the Finch family portrayed in the book but that Burroughs had fabricated or exaggerated various descriptions of their activities.   > It's still a memoir, it's marketed as a memoir, they've agreed one hundred percent that it is a memoir.   The case was later settled with Sony Pictures Entertainment in October 2006, prior to the release of the film adaptation. Burroughs and his publisher, St. Martin's Press, settled with the Turcotte family in August 2007. The Turcottes were reportedly seeking damages of $2 million for invasion of privacy, defamation, and emotional distress; the Turcottes alleged Running with Scissors was largely fictional and written in a sensational manner. Burroughs defended his work as \"entirely accurate\", but agreed to call the work a \"book\" (instead of a \"memoir\") in the author's note, to alter the acknowledgments page in future editions to recognize the Turcotte family's conflicting memories of described events, and express regret for \"any unintentional harm\" to the Turcotte family. Burroughs felt vindicated by the settlement. \"I'm not at all sorry that I wrote [the book]. And you know, the suit settled\u2014it settled in my favor. I didn't change a word of the memoir, not one word of it. It's still a memoir, it's marketed as a memoir, they've agreed one hundred percent that it is a memoir\".   Future printings of Running with Scissors will contain modified language in the Author's Note and Acknowledgments pages. Where the Acknowledgments page had read: \"Additionally, I would like to thank each and every member of a certain Massachusetts family for taking me into their home and accepting me as one of their own,\" the following was substituted: \"Additionally, I would like to thank the real-life members of the family portrayed in this book for taking me into their home and accepting me as one of their own. I recognize that their memories of the events described in this book are different than my own. They are each fine, decent, and hard-working people. The book was not intended to hurt the family. Both my publisher and I regret any unintentional harm resulting from the publishing and marketing of Running with Scissors\"
    Revised claim: running with scissors is somewhat based on the recollections of part of the author's life

    Claim: you can drink at any age in wisconsin.
    Evidence: The drinking age in Wisconsin is 21. Those under the legal drinking age may be served, possess, or consume alcohol if they are with a parent, legal guardian, or spouse who is of legal drinking age. Those age 18 to 20 may also possess (but not consume) alcohol as part of their employment. In the early 70s the sale of alcohol was reduced to the age of18. The 1983 Wisconsin Act 74, effective July 1, 1984, created a drinking age of19. Meeting in special session at the call of the governor, the legislature enacted 1985 Wisconsin Act 337, which raised the drinking age to 21 and brought the state into compliance with the NMDA (National Minimum Drinking Age) on September 1, 1986.   The NMDA law was amended to permit an exception for those persons who were between ages 18 and 21 on the effective date of the law. Wisconsin 19- and 20-year-olds were \u201cgrandfathered in\u201d by this exception after enactment of Act 337. In effect, the state did not have a uniform age of 21 until September 1, 1988.
    Revised claim: you can drink at any age in wisconsin with someone who is of legal drinking age.

    Claim: it is normal for your second toe to be longer.
    Evidence: Morton's toe is the condition of having a first metatarsal which is short in relation to the second metatarsal (see diagram). It is a type of brachymetatarsia.   The distal metatarsal bones vary in relative length compared to the proximal. For most feet, a smooth curve can be traced through the joints at the bases of the toes (the metatarsal-phalangeal, or MTP, joints). But in Morton's foot, the line has to bend more sharply to go through the base of the big toe, as shown in the diagram. This is because the first metatarsal, behind the big toe, is short compared to the second metatarsal, next to it. The longer second metatarsal puts the MTP joint at the base of the second toe further forward.   If the big toe and the second toe are the same length (as measured from the MTP joint to the tip, including only the toe bones or phalanges), then the second toe will protrude farther than the big toe, as shown in the photo.
    Revised claim: your second toe can be longer than your big toe.

    Claim: baby sign language is the same as regular sign language.
    Evidence: Baby sign involves enhanced gestures and altered signs that infants are taught in conjunction with spoken words with the intention of creating richer parent-child communication. The main reason that parents use baby sign is with hope that it will reduce the frustration involved in trying to interpret their pre-verbal child's needs. It can be considered a useful method of communication in the early developmental stages, since speech production follows children's ability to express themselves through bodily movement.   Baby sign is distinct from sign language. Baby sign is used by hearing parents with hearing children to improve communication. Sign languages, including ASL, BSL, ISL and others, are natural languages, typically used in the Deaf community. Sign languages maintain their own grammar, and sentence structure. Because sign languages are as complex to learn as any spoken language, simplified signs are often used with infants in baby sign. Teaching baby signs allows for greater flexibility in the form of sign and does not require the parent to learn the grammar of a sign language. Baby signs are usually gestures or signs taken from the sign language community and modified to make them easier for an infant to form.
    Revised claim: baby sign language is distinct from regular sign language.

    '''
    lm += f'Question: {question}\n'
    # Generate until end of sentence
    lm += 'Revised claim: ' + gen(stop="\n")
    return lm


@guidance
def disambiguation_zero_shot(lm, question):
    lm += '''\
    Please make the following claim less ambiguous with regard to the following evidence.
    '''
    lm += f'Question: {question}\n'
    # Generate until end of sentence
    lm += 'Revised claim: ' + gen(stop="\n")
    return lm

zero_shot_generations = []
four_shot_generations = []
ambiguous_four_shot_generations = []
eight_shot_generations = []
ambiguous_eight_shot_generations = []

"""
for x,sample in enumerate(dis_test_dataset): # or dev
    print("test", x)
    source = sample["source"].replace("\n", " ")
    evidence = sample["evidence"].replace("\n", " ")
    inp = "Claim: "+source+"\n"+"Evidence: "+evidence+"\n"
    zero_shot_generations.append(str(llama3 + disambiguation_zero_shot(inp)).split("Revised claim: ")[-1].strip())
    four_shot_generations.append(str(llama3 + disambiguation_four_shot(inp)).split("Revised claim: ")[-1].strip())
    eight_shot_generations.append(str(llama3 + disambiguation_eight_shot(inp)).split("Revised claim: ")[-1].strip())
    ambiguous_four_shot_generations.append(str(llama3 + disambiguation_ambiguous_four_shot(inp)).split("Revised claim: ")[-1].strip())
    ambiguous_eight_shot_generations.append(str(llama3 + disambiguation_ambiguous_eight_shot(inp)).split("Revised claim: ")[-1].strip())

# write the zero_shot_generations and few_shot_generations out into files with newlines separating items
with open("llm_generations_test/zero_shot_generations.txt", "w") as f:
    f.write("\n".join(zero_shot_generations))
with open("llm_generations_test/four_shot_generations.txt", "w") as f:
    f.write("\n".join(four_shot_generations))
with open("llm_generations_test/eight_shot_generations.txt", "w") as f:
    f.write("\n".join(eight_shot_generations))
with open("llm_generations_test/ambiguous_four_shot_generations.txt", "w") as f:
    f.write("\n".join(ambiguous_four_shot_generations))
with open("llm_generations_test/ambiguous_eight_shot_generations.txt", "w") as f:
    f.write("\n".join(ambiguous_eight_shot_generations))
"""

zero_shot_generations = []
four_shot_generations = []
ambiguous_four_shot_generations = []
eight_shot_generations = []
ambiguous_eight_shot_generations = []

for y,sample in enumerate(dis_dev_dataset):
    print("dev", y)
    source = sample["source"].replace("\n", " ")
    evidence = sample["evidence"].replace("\n", " ")
    inp = "Claim: "+source+"\n"+"Evidence: "+evidence+"\n"
    zero_shot_generations.append(str(llama3 + disambiguation_zero_shot(inp)).split("Revised claim: ")[-1].strip())
    four_shot_generations.append(str(llama3 + disambiguation_four_shot(inp)).split("Revised claim: ")[-1].strip())
    eight_shot_generations.append(str(llama3 + disambiguation_eight_shot(inp)).split("Revised claim: ")[-1].strip())
    ambiguous_four_shot_generations.append(str(llama3 + disambiguation_ambiguous_four_shot(inp)).split("Revised claim: ")[-1].strip())
    ambiguous_eight_shot_generations.append(str(llama3 + disambiguation_ambiguous_eight_shot(inp)).split("Revised claim: ")[-1].strip())

# write the zero_shot_generations and few_shot_generations out into files with newlines separating items
with open("llm_generations/zero_shot_generations_.txt", "w") as f:
    f.write("\n".join(zero_shot_generations))
import pdb; pdb.set_trace()
with open("llm_generations/four_shot_generations.txt", "w") as f:
    f.write("\n".join(four_shot_generations))
with open("llm_generations/eight_shot_generations.txt", "w") as f:
    f.write("\n".join(eight_shot_generations))
with open("llm_generations/ambiguous_four_shot_generations.txt", "w") as f:
    f.write("\n".join(ambiguous_four_shot_generations))
with open("llm_generations/ambiguous_eight_shot_generations.txt", "w") as f:
    f.write("\n".join(ambiguous_eight_shot_generations))
