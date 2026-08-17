define void @labels(i1 %condition) {
entry:
  br i1 %condition, label %short, label %0

short:
  br label %"quoted label"

0:
  br label %"quoted label"

"quoted label":
  br label %this_label_is_longer_than_the_predecessor_comment_column

this_label_is_longer_than_the_predecessor_comment_column:
  ret void
}
