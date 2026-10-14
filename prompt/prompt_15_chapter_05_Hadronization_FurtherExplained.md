
Please implement the following in chapter/chapter_05_children_getting_married.tex:


General comment:

When we write/show for the first time a hadron, please write/ mention it's quark content in brackets.


Specific comments:

(1) Draw the sketch using tikz which visualises the chain of mesons and baryaons before the paragraph "A picture to carry: the wedding" of chapter/chapter_05_children_getting_married.tex

(2) Next in the begining of the section "From partons to hadrons: the Lund string", briefly describe which combinations of quarks/antiquarks are allowed from QCD and which is forbidden. The idea is to illustrate the mesons, baryons, tetraquarks, etc. We do not have to qo into too much details. Maybe an overview will help. Then the rest of the theory can follow as we currently have.


(3) The section "A rope between the dancers: the linear potential and the area law" needs more illustrative sketches to visualise the string breaking, vertices, and connected hadrons. Also briefly explain light-cone coordinates with some sketch which would be helpful in understanding the "the world sheet breaks".


(4) In the begining of the section "How the rope snaps: tunnelling, flavours and transverse momentum" maybe we can show some sketch something along the following lines:
####-----------
\\[ b \\; \underbrace{\longrulefill}\_{\text{string}}\\; \bar u \\]

Suppose five mesons are formed. One possible sequence is:

Initial string

\\[ b \longleftrightarrow \bar u \\]

First string break: \\(u\bar u\\)

\\[ (b\bar u) + (u\longleftrightarrow\bar u) \\]

The original \\(b\\) is already inside the first meson, \\(B^-\\).

Second break: \\(d\bar d\\)

\\[ B^-+(u\bar d)+(d\longleftrightarrow\bar u) \\]

Third break: \\(u\bar u\\)

\\[ B^-+(u\bar d)+(d\bar u)+(u\longleftrightarrow\bar u) \\]

Fourth break: \\(d\bar d\\)

\\[ B^-+(u\bar d)+(d\bar u)+(u\bar d)+(d\bar u) \\]

The final meson contains the original \\(\bar u\\).

So the final chain contains five mesons from four string breaks:

\\[ \boxed{ \underbrace{(b\bar u)}\_{\text{original }b} + (u\bar d) + (d\bar u) + (u\bar d) + \underbrace{(d\bar u)}\_{\text{original }\bar u} } \\]

The key physics:

- The original \\(b\\) is incorporated into the first meson at the \\(b\\)-end.
- The original \\(\bar u\\) is incorporated into the last meson at the opposite end.
- Each intermediate meson contains quarks created through string breaking.
- The fragmentation can proceed from both ends, and the remaining string pieces eventually join to form the final hadron(s).
####-----------

You do not have to show it as it is. This is for example. Something along these lines so that it becomes viually more clear.


(5) "The numbers" paragraph of section "Gluons as kinks: regions of the world sheet" needs some sketch for a better visualisation



Finally, please include similar sketches in the  beginning of example/Had/Standalone/animate_standalone_hadronisation.py
so that the video becomes more effective and illustrative of the hadronisation process.
