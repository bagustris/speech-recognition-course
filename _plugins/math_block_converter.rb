Jekyll::Hooks.register [:pages, :posts], :pre_render do |doc|
    doc.content.gsub!(/```math\n(.*?)```/m, '$$\1$$')
  end
  