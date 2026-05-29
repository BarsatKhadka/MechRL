# Task: induction

- **Model**: gpt2 (n_layers=12, n_heads=12)
- **Batch size**: 20
- **Sequence length**: 49
- **Has corrupted prompts**: True
- **Metric aggregate (clean)**: -0.4675
- **Metric aggregate (corrupted)**: -6.1568

Metadata:
```json
{
  "source": "Olsson 2022 style; real text from wikitext-2",
  "half_len": 25,
  "seq_len": 49
}
```

## Examples

Showing first 20 examples. Note that for IOI the metric is negated logit-diff (lower = better, model prefers correct). For greater-than the metric is negated probability difference between valid-year and invalid-year continuations.

### Example 0

- **CLEAN**:     ` near Arkadelphia on the Ouachita River , 75 miles south from Little Rock . The tools , machinery , and the material near Arkadelphia on the Ouachita River , 75 miles south from Little Rock . The tools , machinery , and the`
- **CORRUPTED**: ` last part of the month . "The military force at Little Rock under Dunnington 's command consisted of four officers : himself near Arkadelphia on the Ouachita River , 75 miles south from Little Rock . The tools , machinery , and the`
- **Difference**: 25/49 tokens differ at positions 0, 1, 2, 3, 4 (+20 more); clean='near'|'Ark'|'adelphia'|'on'|'the' vs corrupted='last'|'part'|'of'|'the'|'month'

### Example 1

- **CLEAN**:     ` map : once a character is selected , the player moves the character around the battlefield in third @-@ person . A character map : once a character is selected , the player moves the character around the battlefield in third @-@ person . A`
- **CORRUPTED**: ` , it was part of Little Rock 's first military installation . Since its decommissioning , The Tower Building has housed two map : once a character is selected , the player moves the character around the battlefield in third @-@ person . A`
- **Difference**: 25/49 tokens differ at positions 0, 1, 2, 3, 4 (+20 more); clean='map'|':'|'once'|'a'|'character' vs corrupted=','|'it'|'was'|'part'|'of'

### Example 2

- **CLEAN**:     ` Hall . The museum remained in the tower building for approximately fifty @-@ five years . The area surrounding the Tower Building had Hall . The museum remained in the tower building for approximately fifty @-@ five years . The area surrounding the Tower Building`
- **CORRUPTED**: ` at The Chicago World 's Fair . At the fair 's conclusion , artifacts from the exhibit were displayed in the Tower Building Hall . The museum remained in the tower building for approximately fifty @-@ five years . The area surrounding the Tower Building`
- **Difference**: 25/49 tokens differ at positions 0, 1, 2, 3, 4 (+20 more); clean='Hall'|'.'|'The'|'museum'|'remained' vs corrupted='at'|'The'|'Chicago'|'World'|"'"

### Example 3

- **CLEAN**:     ` for a suitable person to head this activity , General Hindman turned to the Confederate Navy and borrowed Lieutenant John W. Dunnington for a suitable person to head this activity , General Hindman turned to the Confederate Navy and borrowed Lieutenant John W. Dun`
- **CORRUPTED**: ` Indian nation have all been recalled from winter quarters to reinforce the garrison at Fort Smith . The garrison at Fort Smith had been previously for a suitable person to head this activity , General Hindman turned to the Confederate Navy and borrowed Lieutenant John W. Dun`
- **Difference**: 25/49 tokens differ at positions 0, 1, 2, 3, 4 (+20 more); clean='for'|'a'|'suitable'|'person'|'to' vs corrupted='Indian'|'nation'|'have'|'all'|'been'

### Example 4

- **CLEAN**:     ` exclusively devoted to the uses and purposes of a public park " for 1 @,@ 000 acres ( 4 km ² ) exclusively devoted to the uses and purposes of a public park " for 1 @,@ 000 acres ( 4 km ²`
- **CORRUPTED**: ` were only 1 @,@ 364 percussion weapons available . Disposition of the weapons found in the Arsenal is somewhat sketchy , exclusively devoted to the uses and purposes of a public park " for 1 @,@ 000 acres ( 4 km ²`
- **Difference**: 25/49 tokens differ at positions 0, 1, 2, 3, 4 (+20 more); clean='exclusively'|'devoted'|'to'|'the'|'uses' vs corrupted='were'|'only'|'1'|'@'|','

### Example 5

- **CLEAN**:     `. Edmund 's and St. Andrew 's in Croydon – " low " churches for the less privileged . Barker. Edmund 's and St. Andrew 's in Croydon – " low " churches for the less privileged .`
- **CORRUPTED**: `Kurt and Riela were featured in the Nintendo 3DS crossover Project X Zone , representing the Valkyria series .. Edmund 's and St. Andrew 's in Croydon – " low " churches for the less privileged .`
- **Difference**: 25/49 tokens differ at positions 0, 1, 2, 3, 4 (+20 more); clean='.'|'Edmund'|"'"|'s'|'and' vs corrupted='K'|'urt'|'and'|'R'|'iel'

### Example 6

- **CLEAN**:     ` Tower Building .The arsenal was listed in the National Register of Historic Places in 1970 . Due to its association with the Camden Expedition Tower Building .The arsenal was listed in the National Register of Historic Places in 1970 . Due to its association with the Camden`
- **CORRUPTED**: ` player could identify while maintaining a sense of reality within the Valkyria Chronicles world . The main color of the Nameless was Tower Building .The arsenal was listed in the National Register of Historic Places in 1970 . Due to its association with the Camden`
- **Difference**: 24/49 tokens differ at positions 0, 1, 2, 3, 4 (+19 more); clean='Tower'|'Building'|'.'|'The'|'arsenal' vs corrupted='player'|'could'|'identify'|'while'|'maintaining'

### Example 7

- **CLEAN**:     ` be built of stone , however , masonry was used instead . The Arkansas Gazette referred to the structure as " A splendid specimen be built of stone , however , masonry was used instead . The Arkansas Gazette referred to the structure as " A splendid`
- **CORRUPTED**: ` reality within the Valkyria Chronicles world . The main color of the Nameless was black . As with the previous Valky be built of stone , however , masonry was used instead . The Arkansas Gazette referred to the structure as " A splendid`
- **Difference**: 25/49 tokens differ at positions 0, 1, 2, 3, 4 (+20 more); clean='be'|'built'|'of'|'stone'|',' vs corrupted='reality'|'within'|'the'|'V'|'alky'

### Example 8

- **CLEAN**:     `ian Army rather than serve as an anonymous shadow force . This is short @-@ lived , however , as following Maximilian Army rather than serve as an anonymous shadow force . This is short @-@ lived , however , as following Max`
- **CORRUPTED**: ` Light ) -Tomoshibi- " ( 灯 @-@ TOMOSHIBI- ) , was sungian Army rather than serve as an anonymous shadow force . This is short @-@ lived , however , as following Max`
- **Difference**: 25/49 tokens differ at positions 0, 1, 2, 3, 4 (+20 more); clean='ian'|'Army'|'rather'|'than'|'serve' vs corrupted='Light'|')'|'-'|'Tom'|'os'

### Example 9

- **CLEAN**:     ` St. Edmund 's Church , Pitlake , and her painting of the Christ Child , The Darling of the World Has Come St. Edmund 's Church , Pitlake , and her painting of the Christ Child , The Darling of the World Has`
- **CORRUPTED**: ` , which she did .Barker continued to attend evening classes at the Croydon Art School between the 1920s and St. Edmund 's Church , Pitlake , and her painting of the Christ Child , The Darling of the World Has`
- **Difference**: 23/49 tokens differ at positions 0, 1, 2, 3, 4 (+18 more); clean='St'|'.'|'Edmund'|"'"|'s' vs corrupted=','|'which'|'she'|'did'|'.'

### Example 10

- **CLEAN**:     ` theme he designed around his original vision of the project was rejected . He redid the main theme about seven times through the music theme he designed around his original vision of the project was rejected . He redid the main theme about seven times through the`
- **CORRUPTED**: ` sentiment . The rumor was interpreted by some Arkansans as a call from the governor to assemble to help expel the federal troops theme he designed around his original vision of the project was rejected . He redid the main theme about seven times through the`
- **Difference**: 25/49 tokens differ at positions 0, 1, 2, 3, 4 (+20 more); clean='theme'|'he'|'designed'|'around'|'his' vs corrupted='sentiment'|'.'|'The'|'rumor'|'was'

### Example 11

- **CLEAN**:     ` or the chance to prove themselves in the war with Gallia , it is Dahau 's last trump card in creating a or the chance to prove themselves in the war with Gallia , it is Dahau 's last trump card in creating`
- **CORRUPTED**: ` a seemingly jinxed young woman who is unknowingly a descendant of the Valkyria . Together with their fellow squad members or the chance to prove themselves in the war with Gallia , it is Dahau 's last trump card in creating`
- **Difference**: 25/49 tokens differ at positions 0, 1, 2, 3, 4 (+20 more); clean='or'|'the'|'chance'|'to'|'prove' vs corrupted='a'|'seemingly'|'j'|'inx'|'ed'

### Example 12

- **CLEAN**:     ` no instructions from his superiors and was forced to withdraw his troops . He agreed to surrender the arsenal as long as the governor agreed no instructions from his superiors and was forced to withdraw his troops . He agreed to surrender the arsenal as long as the governor`
- **CORRUPTED**: ` The arsenal commander received word from Washington that the Little Rock site must be abandoned " not later than October 1 , 1890 . " no instructions from his superiors and was forced to withdraw his troops . He agreed to surrender the arsenal as long as the governor`
- **Difference**: 25/49 tokens differ at positions 0, 1, 2, 3, 4 (+20 more); clean='no'|'instructions'|'from'|'his'|'superiors' vs corrupted='The'|'arsenal'|'commander'|'received'|'word'

### Example 13

- **CLEAN**:     ` Advertiser remarked , " Her drawings show a remarkable freedom of spirit . She has distinct promise . "Following her father � Advertiser remarked , " Her drawings show a remarkable freedom of spirit . She has distinct promise . "Following her father`
- **CORRUPTED**: ` As an armed Gallian force invading the Empire just following the two nations ' cease @-@ fire would certainly wreck their newfound Advertiser remarked , " Her drawings show a remarkable freedom of spirit . She has distinct promise . "Following her father`
- **Difference**: 25/49 tokens differ at positions 0, 1, 2, 3, 4 (+20 more); clean='Ad'|'vertis'|'er'|'remarked'|',' vs corrupted='As'|'an'|'armed'|'Gall'|'ian'

### Example 14

- **CLEAN**:     `M1841 rifle ( " Mississippi Rifle " ) 54M1847 musketoon 2Hall 's carbines 267HallM1841 rifle ( " Mississippi Rifle " ) 54M1847 musketoon 2Hall 's carbines 267`
- **CORRUPTED**: ` in charge of the Arsenal . Dunnington presumably returned to his naval duties and the Ponchartrain .A " Summary of theM1841 rifle ( " Mississippi Rifle " ) 54M1847 musketoon 2Hall 's carbines 267`
- **Difference**: 25/49 tokens differ at positions 0, 1, 2, 3, 4 (+20 more); clean='M'|'18'|'41'|'rifle'|'(' vs corrupted='in'|'charge'|'of'|'the'|'Arsenal'

### Example 15

- **CLEAN**:     ` the fact that the military had assembled believing they were following his orders and the consensus of the citizens of Little Rock against any armed the fact that the military had assembled believing they were following his orders and the consensus of the citizens of Little Rock against any`
- **CORRUPTED**: ` Dorothy and her sister collaborated upon only two books : Our Darling 's First Book and the Christian @-@ themed , He the fact that the military had assembled believing they were following his orders and the consensus of the citizens of Little Rock against any`
- **Difference**: 24/49 tokens differ at positions 0, 1, 2, 3, 4 (+19 more); clean='the'|'fact'|'that'|'the'|'military' vs corrupted='Dorothy'|'and'|'her'|'sister'|'collaborated'

### Example 16

- **CLEAN**:     ` continued quality and the tweaks to balance and content . Its one major criticism were multiple difficulty spikes , something that had affected the previous continued quality and the tweaks to balance and content . Its one major criticism were multiple difficulty spikes , something that had affected the`
- **CORRUPTED**: ` took correspondence courses in art , probably until about 1919 . In 1908 at 13 years , she entered an evening class at the C continued quality and the tweaks to balance and content . Its one major criticism were multiple difficulty spikes , something that had affected the`
- **Difference**: 24/49 tokens differ at positions 0, 1, 2, 3, 4 (+19 more); clean='continued'|'quality'|'and'|'the'|'tweaks' vs corrupted='took'|'correspondence'|'courses'|'in'|'art'

### Example 17

- **CLEAN**:     ` of a public park " for 1 @,@ 000 acres ( 4 km ² ) in Big Rock Mountain on the north of a public park " for 1 @,@ 000 acres ( 4 km ² ) in Big Rock Mountain on the`
- **CORRUPTED**: ` received and mostly issued to troops in service .Repaired and painted :4 gun carriagesPerformed :Guard , office , of a public park " for 1 @,@ 000 acres ( 4 km ² ) in Big Rock Mountain on the`
- **Difference**: 25/49 tokens differ at positions 0, 1, 2, 3, 4 (+20 more); clean='of'|'a'|'public'|'park'|'"' vs corrupted='received'|'and'|'mostly'|'issued'|'to'

### Example 18

- **CLEAN**:     ` servicemen and servicewomen of the United States and commemorate the birthplace of Douglas MacArthur .= Cicely Mary Barker = servicemen and servicewomen of the United States and commemorate the birthplace of Douglas MacArthur .= Cicely Mary Barker`
- **CORRUPTED**: ` museums . It was home to the Arkansas Museum of Natural History and Antiquities from 1942 to 1997 and the MacArthur Museum of Arkansas servicemen and servicewomen of the United States and commemorate the birthplace of Douglas MacArthur .= Cicely Mary Barker`
- **Difference**: 25/49 tokens differ at positions 0, 1, 2, 3, 4 (+20 more); clean='servic'|'emen'|'and'|'servic'|'ew' vs corrupted='museums'|'.'|'It'|'was'|'home'

### Example 19

- **CLEAN**:     ` from early games ' method of distributing to different unit types .= = Plot = =The game takes place during the Second Euro from early games ' method of distributing to different unit types .= = Plot = =The game takes place during the Second`
- **CORRUPTED**: ` meals for Barker were hired . She spent much time in bed at home amusing herself with painting books and a nursery library that included from early games ' method of distributing to different unit types .= = Plot = =The game takes place during the Second`
- **Difference**: 25/49 tokens differ at positions 0, 1, 2, 3, 4 (+20 more); clean='from'|'early'|'games'|"'"|'method' vs corrupted='meals'|'for'|'Barker'|'were'|'hired'
