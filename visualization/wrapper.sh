vid=$1
tag=$2

basename "$vid"
f="$(basename -- $vid)"
ff="$(dirname -- $vid)"
speaker="$(basename -- $ff)"

bash run_visualize.sh $f $ff 500 $speaker $tag
