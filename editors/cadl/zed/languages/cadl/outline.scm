(flow_declaration
  ["flow" "rtype"] @context
  name: (identifier) @name) @item

(allo_declaration
  "allo" @context
  name: (identifier) @name) @item

(static_declaration
  "static" @context
  name: (identifier) @name) @item

(register_declaration
  "register" @context
  name: (identifier) @name) @item

(regfile_declaration
  "regfile" @context
  name: (identifier) @name) @item
